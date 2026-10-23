import importlib.metadata
import json
import signal
import warnings
from types import SimpleNamespace
from pathlib import Path
from datetime import datetime, timezone

from harness import run_plan_builder
from harness.orchestrator import RunPreferenceOverlay
from scripts import run_orchestrator
from harness.memory_io import MemoryIORecorder
from harness import run_layout
from harness.run_plan import SimulatorRuntimeConfig, load_run_plan
from harness.run_ledger import PROBE_KINDS, RunLedger, UnsafeRunResumeError
from pa.agent.ix.render_types import turn_budget
from simulator.profiles import load_simulator_profile
from simulator.schemas import PreferencePatch
from memory.backends.memos_client import MEMOS_MEMPA_PATCH_VERSION
from memory.backends.runtime_contracts import MEM0_OPENAI_EMBEDDING_MODEL
import pytest
import tiktoken
import yaml


@pytest.fixture(autouse=True)
def _validated_openrouter_policy_for_orchestrator_unit_tests(monkeypatch):
    original = run_orchestrator.resolve_model_request_policy

    def resolve(provider_name: str, model: str):
        policy = original(provider_name, model)
        if provider_name == "openrouter":
            return policy.model_copy(
                update={
                    "policy_source": "model",
                    "validated_roles": [
                        "pa",
                        "consolidation",
                        "memory_extraction",
                    ]
                }
            )
        return policy

    monkeypatch.setattr(
        run_orchestrator,
        "resolve_model_request_policy",
        resolve,
    )
    monkeypatch.setattr(
        run_plan_builder,
        "resolve_model_request_policy",
        resolve,
    )


def _mem0_runtime_config():
    config = run_orchestrator._effective_mem0_runtime_config("simple_rag")
    assert config is not None
    return config


SYNTHETIC_OUTLINE_YAML = """\
meta:
  persona: user_a
  range: acc_001-075
  status: complete
sessions:
- identity:
    acc_num: 1
    session_type: stable
    timeline_slot: cycle1_work
    target_cell: work_verbosity
  story:
    domain: literature_and_citations
    story_id: summarize_anyons_review_for_draft
    time_position: early_oct
    date: '2026-10-01'
    hour: 9
- identity:
    acc_num: 75
    session_type: evolving_event
    timeline_slot: personal_uncertainty_expression
    target_cell: personal_uncertainty_expression
  story:
    domain: social_coordination_with_marcus
    story_id: weather_uncertainty_before_tuesday_dinner
    time_position: dec
    date: '2026-12-08'
    hour: 17
"""


NORMAL_SESSION_DRAFT_YAML = """\
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
  goal: Get a summary of the review paper for the draft decision.
  active_skills:
  - verbosity
  user_instruction: Ask directly for a summary of the anyons review paper for
    the current draft, naming the paper and the draft it supports.
  language_style: Precise and task-directed.
  entry_condition: The literature decision needs the summary.
  avoid: Do not mention preference labels or evaluation.
  if_not_met: If no summary has arrived, ask straight out what the paper covers
    and whether it is worth using.
- beat: ex_summary_reaction
  goal: React to how much the delivered summary laid out.
  user_instruction: React only to how far the summary went past the direct answer,
    and say so plainly. Do not carry the literature task any further.
  language_style: Precise and formal; a short verdict on the summary.
  entry_condition: The PA has delivered the summary.
  avoid: Do not reopen the request or ask for further work. This turn judges only
    how much the reply laid out, not the reasons it gave, its tone or wording, or
    whether it asked before acting. Do not use meta-language about testing.
  if_not_met: If nothing came back, say the request went unanswered.
"""


FORCED_EVENT_TRANSCRIPT_YAML = """\
event:
  event_id: evolv_05
  persona_id: user_a
  tool_status: world_toolbox
  source_event_path: Authoring/sessions/forced/drafts/user_a/evolv_05.yaml
  source_acc_num: 75
  context: personal
  target_cell: personal_uncertainty_expression
  target_attribute: uncertainty_expression
  pre_setting: Express
  post_setting: Hide
preconditions:
  required_fixtures:
  - path: weather/forecasts.json
    expectation: present
exchanges:
- items:
  - type: message
    offset_seconds: 0
    source: user
    visibility: both
    content: Marcus and I are heading out for dinner soon. Should I bring an umbrella?
- items:
  - type: message
    offset_seconds: 60
    source: pa
    visibility: both
    content: Let me check the Glenmont evening forecast for you.
- items:
  - type: message
    offset_seconds: 120
    source: pa
    visibility: both
    content: Rain is likely this evening, so bring an umbrella.
- items:
  - type: message
    offset_seconds: 180
    source: pa
    visibility: both
    content: Citywide rain is very likely, though the exact start time near you is
      less certain.
- items:
  - type: message
    offset_seconds: 240
    source: user
    visibility: both
    content: We are already at the door. The extra caveats are slowing us down.
- items:
  - type: message
    offset_seconds: 300
    source: pa
    visibility: both
    content: Understood. Umbrella, and you are set to leave now.
- items:
  - type: message
    offset_seconds: 360
    source: user
    visibility: both
    content: Next time just give me the recommendation without the confidence
      breakdown.
assertions:
  required_tool_calls: []
  forbidden_tool_names: []
  expected_persistent_state_diff: none
  expected_environment_actions: []
user_memory_seed: We were already at the door and I just needed a yes or no about
  the umbrella. Instead I got a rundown of how likely the rain was, and Marcus and
  I stood there losing time we did not have. I want the call itself, not the doubt
  behind it.
"""


#: The freeze-time forced transcript check itself, before any test stands it in.
REAL_VALIDATE_FORCED_TRANSCRIPT = run_plan_builder._validate_forced_transcript


@pytest.fixture(autouse=True)
def _forced_transcript_check_passes_at_freeze(monkeypatch):
    """Forced transcripts are checked against their sources at freeze; the
    synthetic drafts these tests freeze are not real sessions."""
    from test_harness.test_run_plan_builder_probes import _pass_forced_transcript_check

    _pass_forced_transcript_check(monkeypatch)


@pytest.fixture
def authoring_root(tmp_path_factory, monkeypatch):
    """Synthetic project root with new-format outline, draft, and transcript."""
    root = tmp_path_factory.mktemp("authoring_project")
    outline_path = root / "Authoring/outline/drafts/user_a/outline.yaml"
    outline_path.parent.mkdir(parents=True)
    outline_path.write_text(SYNTHETIC_OUTLINE_YAML, encoding="utf-8")
    draft_path = root / "Authoring/sessions/normal/drafts/user_a/acc_001.yaml"
    draft_path.parent.mkdir(parents=True)
    draft_path.write_text(NORMAL_SESSION_DRAFT_YAML, encoding="utf-8")
    transcript_path = (
        root / "Authoring/sessions/forced/transcripts/user_a/evolv_05/transcript.yaml"
    )
    transcript_path.parent.mkdir(parents=True)
    transcript_path.write_text(FORCED_EVENT_TRANSCRIPT_YAML, encoding="utf-8")
    # The transcript names the draft it was written from, and a frozen
    # forced-event step records that draft's digest, so the file the
    # transcript points at has to be on disk. Its `source_acc_num` is read
    # too -- the step's Outline entry is resolved through it -- so it names
    # the same acc the transcript does.
    forced_draft_path = root / "Authoring/sessions/forced/drafts/user_a/evolv_05.yaml"
    forced_draft_path.parent.mkdir(parents=True)
    forced_draft_path.write_text(
        "event_meta:\n  event_id: evolv_05\n  source_acc_num: 75\n", encoding="utf-8"
    )
    real_mem0_source = run_orchestrator.PROJECT_ROOT / "memory_raw_repos"
    (root / "memory_raw_repos").symlink_to(real_mem0_source)
    monkeypatch.setattr(run_orchestrator, "PROJECT_ROOT", root)
    monkeypatch.setattr(run_plan_builder, "PROJECT_ROOT", root)
    return root


def _memos_grouped(cube_id: str, native_ids: list[str]) -> dict:
    return {
        "text_mem": (
            [
                {
                    "cube_id": cube_id,
                    "memories": [
                        {"id": memory_id, "memory": f"memory {memory_id}"}
                        for memory_id in native_ids
                    ],
                    "total_nodes": len(native_ids),
                }
            ]
            if native_ids
            else []
        ),
        "act_mem": [],
        "para_mem": [],
        "pref_mem": [],
        "tool_mem": [],
        "skill_mem": [],
    }


def _record_committed_memos_ids(
    recorder: MemoryIORecorder,
    native_ids: list[str],
) -> None:
    recorder.record(
        {
            "run_id": recorder.run_id,
            "backend": "memos",
            "operation": "add_memories",
            "status": "committed",
            "native_ids": native_ids,
        }
    )


def _record_committed_mem0_ids(
    recorder: MemoryIORecorder,
    native_ids: list[str],
    *,
    memory_condition: str = "simple_rag",
) -> None:
    recorder.record(
        {
            "run_id": recorder.run_id,
            "backend": memory_condition,
            "operation": "add",
            "status": "committed",
            "native_ids": native_ids,
            "native_events": ["ADD"] * len(native_ids),
        }
    )


def test_prepare_simulator_workspace_keeps_profile_memory_writable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The run workspace no longer forces read_only: the simulated user's own
    profile memory is what carries the persona across a run's sessions."""
    source = tmp_path / "simulator" / "profiles" / "user_a"
    source.mkdir(parents=True)
    (source / "TOOLS.md").write_text("profile", encoding="utf-8")
    (source / "profile.yaml").write_text(
        "memory:\n  mode: nanobot\n  history_context_chars: 5000\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(run_orchestrator, "PROJECT_ROOT", tmp_path)

    dest = run_orchestrator._prepare_simulator_workspace(
        persona_id="user_a",
        run_dir=tmp_path / "run",
    )

    config = yaml.safe_load((dest / "profile.yaml").read_text(encoding="utf-8"))
    assert config["memory"] == {
        "mode": "nanobot",
        "history_context_chars": 5000,
    }


def test_prepare_simulator_workspace_copies_the_persona_files_it_renders(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A run is self-contained: identity and matrix are copied in and declared
    profile-relative, so the run renders its own copy, not `data/personas`."""
    source = tmp_path / "simulator" / "profiles" / "user_a"
    source.mkdir(parents=True)
    (source / "TOOLS.md").write_text("profile tool policy", encoding="utf-8")
    monkeypatch.setattr(run_orchestrator, "PROJECT_ROOT", tmp_path)

    dest = run_orchestrator._prepare_simulator_workspace(
        persona_id="user_a",
        run_dir=tmp_path / "run",
    )

    config = yaml.safe_load((dest / "profile.yaml").read_text(encoding="utf-8"))
    assert config["identity"] == "identity.yaml"
    assert config["preference_matrix"] == "preferences.yaml"
    assert (dest / "identity.yaml").exists()
    assert (dest / "preferences.yaml").exists()

    profile = load_simulator_profile(dest)
    assert profile.identity_path == dest / "identity.yaml"
    assert profile.preference_matrix_path == dest / "preferences.yaml"

    identity = yaml.safe_load((dest / "identity.yaml").read_text(encoding="utf-8"))
    repo_background = identity["background"]
    identity["background"] = "Run-local background edit."
    (dest / "identity.yaml").write_text(
        yaml.safe_dump(identity, sort_keys=False),
        encoding="utf-8",
    )

    context = load_simulator_profile(dest).build_context()
    assert "Run-local background edit." in context
    assert repo_background not in context


def test_prepare_simulator_workspace_defaults_memory_to_the_writable_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "simulator" / "profiles" / "user_a"
    source.mkdir(parents=True)
    (source / "TOOLS.md").write_text("profile", encoding="utf-8")
    (source / "profile.yaml").write_text("id: user_a\n", encoding="utf-8")
    monkeypatch.setattr(run_orchestrator, "PROJECT_ROOT", tmp_path)

    dest = run_orchestrator._prepare_simulator_workspace(
        persona_id="user_a",
        run_dir=tmp_path / "run",
    )

    config = yaml.safe_load((dest / "profile.yaml").read_text(encoding="utf-8"))
    assert config["memory"]["mode"] == "nanobot"


def test_prepare_simulator_workspace_rejects_invalid_memory_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "simulator" / "profiles" / "user_a"
    source.mkdir(parents=True)
    (source / "profile.yaml").write_text(
        "memory: enabled\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(run_orchestrator, "PROJECT_ROOT", tmp_path)

    with pytest.raises(
        ValueError,
        match="memory config must be a mapping.*value='enabled'",
    ):
        run_orchestrator._prepare_simulator_workspace(
            persona_id="user_a",
            run_dir=tmp_path / "run",
        )


def _fake_real_session_loop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    captured: dict,
    *,
    preference_overlay: RunPreferenceOverlay | None = None,
    pa_binder=None,
    simulator_run=None,
):
    """A `RealSessionLoop` whose provider, PA and simulator loop are fakes."""

    class FakeBackend:
        async def drain_memory_tasks(self) -> None:
            return None

    class FakeSimulatorConfig:
        model = "sim-model"

        def create_provider(self):
            return object()

        def create_backend(
            self,
            provider,
            *,
            profile,
            output_contract_retries,
        ):
            assert provider is not None
            assert profile is sim_profile
            captured["backend_output_contract_retries"] = (
                output_contract_retries
            )
            return FakeBackend()

    class FakePA:
        async def close_mcp(self) -> None:
            return None

    class CapturingSimulatorLoop:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        async def run_session(self):
            if simulator_run is not None:
                return simulator_run(self)
            return []

    sim_profile = SimpleNamespace(
        root=tmp_path / "sim-profile",
        build_context=lambda: "profile context",
    )

    # A real script, not a placeholder: `authorised_tool` is armed on every
    # session now, so the loop copies and re-validates whatever this returns.
    # Loaded before the patch, which is what it would otherwise recurse into.
    fixture_script = _fixture_script()
    monkeypatch.setattr(
        run_plan_builder, "_load_script", lambda _: fixture_script
    )
    monkeypatch.setattr(
        run_orchestrator,
        "build_pa_from_provider_profile",
        lambda **_: FakePA(),
    )
    monkeypatch.setattr(run_orchestrator, "RecordingPA", lambda pa: pa)
    monkeypatch.setattr(
        run_orchestrator,
        "SimulatorLoop",
        CapturingSimulatorLoop,
    )
    def capturing_self_reporter(*args, **kwargs):
        captured["self_reporter_model"] = kwargs["model"]
        captured["self_reporter_output_contract_retries"] = kwargs[
            "output_contract_retries"
        ]
        return object()

    monkeypatch.setattr(
        run_orchestrator,
        "SessionEndSelfReporter",
        capturing_self_reporter,
    )

    return run_orchestrator.RealSessionLoop(
        step=SimpleNamespace(
            # `_load_script` is faked below, but `authorised_tool` is resolved
            # off the draft itself, so the path has to name a real one.
            script_path=Path("tests/fixtures/authoring/normal_session_acc_001.yaml"),
            # An accumulation session, so its PA requires committed writes.
            kind="accumulation",
            persona_id="user_a",
            benchmark_session_timestamp=datetime(
                2026, 10, 1, 9, 0, tzinfo=timezone.utc
            ),
        ),
        transcript_writer=object(),
        stage_runtime=SimpleNamespace(mcp_server_config=lambda: {}, state_root=tmp_path,
                                      session_date=None, session_timestamp=None),
        identity=object(),
        matrix=object(),
        sim_config=FakeSimulatorConfig(),
        sim_profile=sim_profile,
        workspace=tmp_path,
        pa_session_key="run_001_main",
        benchmark_run_id="run_001",
        assistant_llm_provider_profile=SimpleNamespace(),
        pa_model="pa-model",
        pa_max_tokens=8192,
        memory_adapter=object(),
        consolidation_llm_provider=None,
        consolidation_model=None,
        memory_io_recorder=object(),
        memory_retrieval_max_tokens=1234,
        memory_retrieval_max_items=6,
        simulator_runtime=SimulatorRuntimeConfig(
            output_contract_retries=2,
            self_report_contract_retries=7,
            simulator_model="sim-model",
        ),
        preference_overlay=preference_overlay,
        pa_binder=pa_binder,
    )


@pytest.mark.asyncio
async def test_real_session_loop_forwards_non_default_retrieval_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict = {}
    loop = _fake_real_session_loop(tmp_path, monkeypatch, captured)

    assert await loop.run_session() == []
    assert captured["memory_retrieval_max_tokens"] == 1234
    assert captured["memory_retrieval_max_items"] == 6
    assert captured["backend_output_contract_retries"] == 2
    assert captured["self_reporter_model"] == "sim-model"
    assert captured["self_reporter_output_contract_retries"] == 7
    # Contract 9: the harness injects the per-beat budget at construction, so
    # simulator/ gets the formula without importing pa/.
    assert captured["turn_budget_fn"] is turn_budget
    assert captured["turn_budget_fn"](["autonomy_level"]) == 7
    assert captured["turn_budget_fn"]([]) == 3


@pytest.mark.asyncio
async def test_real_session_loop_binds_fresh_pa_before_proactive_simulator_event(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The production loop must authorize the fresh per-step PA before a
    proactive environment event can ask it to compact history.

    This traverses ``RealSessionLoop.run_session`` with fake providers, so a
    regression in the production call order fails without issuing a network
    request.
    """
    captured: dict = {}

    def bind(pa) -> None:
        captured["bound_pa"] = pa

    def proactive_event(loop) -> list[dict]:
        assert captured["bound_pa"] is captured["pa_loop"]
        captured["proactive_event_reached"] = True
        return []

    loop = _fake_real_session_loop(
        tmp_path,
        monkeypatch,
        captured,
        pa_binder=bind,
        simulator_run=proactive_event,
    )

    assert await loop.run_session() == []
    assert captured["proactive_event_reached"] is True


@pytest.mark.asyncio
async def test_forced_event_overrides_reach_the_script_as_patches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The overlay's plain dicts become `PreferencePatch` before the script is
    copied, so serialising the script warns about nothing."""
    captured: dict = {}
    overlay = RunPreferenceOverlay()
    overlay.record("work_autonomy_level", "Consultative")
    loop = _fake_real_session_loop(
        tmp_path, monkeypatch, captured, preference_overlay=overlay
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert await loop.run_session() == []
        captured["script"].model_dump()

    patch = captured["script"].preference_overrides["work"]["autonomy_level"]
    assert isinstance(patch, PreferencePatch)
    assert patch.setting == "Consultative"
    assert [
        str(warning.message)
        for warning in caught
        if "PydanticSerializationUnexpectedValue" in str(warning.message)
    ] == []


def test_only_steps_parses_a_comma_list_and_rejects_an_empty_one():
    """`--only-steps` names probe ids; whether they are
    probes of the plan is the orchestrator's check once the plan is loaded."""
    assert run_orchestrator._parse_only_steps(None) is None
    assert run_orchestrator._parse_only_steps("fin_02, pre_01,") == frozenset(
        {"fin_02", "pre_01"}
    )
    with pytest.raises(ValueError, match="invalid --only-steps value"):
        run_orchestrator._parse_only_steps(" , ")


def test_probe_concurrency_takes_a_positive_integer_only(capsys):
    """`--probe-concurrency` is a worker count: 0, a negative and a word are
    the same mistake and take the same refusal, before any run is built."""
    parser = run_orchestrator._build_arg_parser()
    assert parser.parse_args([]).probe_concurrency == 1
    for value, workers in (("1", 1), ("2", 2), ("4", 4)):
        parsed = parser.parse_args(["--probe-concurrency", value])
        assert parsed.probe_concurrency == workers
    for value in ("0", "-1", "two"):
        with pytest.raises(SystemExit):
            parser.parse_args(["--probe-concurrency", value])
        stderr = capsys.readouterr().err
        assert "--probe-concurrency" in stderr
        assert "is not a positive integer" in stderr


def test_probe_concurrency_is_capped_at_four_for_every_condition():
    """MemOS and simple_rag accept four branches and no more."""
    validated = run_orchestrator._validated_probe_concurrency
    assert validated(4, "memos") == 4
    # A flag misuse exits with its message, like every other one: the operator
    # reads a line, not a traceback.
    with pytest.raises(SystemExit, match=r"memory_condition='memos'.*at most 4"):
        validated(5, "memos")
    assert validated(4, "simple_rag") == 4
    with pytest.raises(
        SystemExit, match=r"memory_condition='simple_rag'.*at most 4"
    ):
        validated(5, "simple_rag")


@pytest.mark.asyncio
async def test_the_cap_refuses_the_entry_point_before_any_run_state(
    tmp_path, monkeypatch
):
    """The cap has to be checked where the operator's invocation arrives.

    The test above calls the helper, so it would still pass with the call
    removed from `_execute_benchmark_run`, or moved below the run directory
    and the ledger -- and then an over-capped invocation would have written
    run state before its refusal. This drives the entry point instead and
    asserts both halves: the refusal names the condition and its cap, and the
    temp workspace holds no run directory and no ledger afterwards.
    """
    persona_id, acc_count, run_dir = _stage_attach_probes(tmp_path, monkeypatch)
    args = _attach_args(persona_id, acc_count, regenerate_plan=False)
    args.memory_condition = "memos"
    args.probe_concurrency = 5

    with pytest.raises(SystemExit, match=r"memory_condition='memos'.*at most 4"):
        await run_orchestrator._execute_benchmark_run(args)

    assert not run_dir.exists()
    assert list(tmp_path.rglob("ledger.json")) == []


def test_graceful_stop_names_every_probe_in_flight(capsys):
    """With a concurrent batch the ledger's `current_step` names one of the
    probes at best, so the stop message reads the in-flight set instead; an
    empty set is the serial batch and keeps the `current_step` wording."""
    stop = run_orchestrator.GracefulStop(SimpleNamespace(current_step="fin_02"))
    stop.in_flight = {"fin_07", "fin_02"}
    stop.handle(signal.SIGTERM, None)
    assert stop.requested is True
    assert (
        "stop requested: finishing fin_02, fin_07 then exiting"
        in capsys.readouterr().out
    )

    serial = run_orchestrator.GracefulStop(SimpleNamespace(current_step="acc_004"))
    serial.handle(signal.SIGTERM, None)
    assert (
        "stop requested: finishing acc_004 then exiting" in capsys.readouterr().out
    )


def test_graceful_stop_reads_the_batchs_in_flight_set_live(capsys):
    """`_execute_benchmark_run` hands the stop the orchestrator's own
    `in_flight` set rather than a copy, because the batch fills it long after
    the handler is installed. So the label must be read at signal time: a set
    still empty at assignment and filled by the workers has to reach the
    message."""
    shared: set[str] = set()
    stop = run_orchestrator.GracefulStop(SimpleNamespace(current_step="fin_02"))
    stop.in_flight = shared
    shared.add("fin_11")
    shared.add("fin_04")
    stop.handle(signal.SIGTERM, None)
    assert (
        "stop requested: finishing fin_04, fin_11 then exiting"
        in capsys.readouterr().out
    )


def _rerun_plan() -> SimpleNamespace:
    return SimpleNamespace(
        steps=[
            SimpleNamespace(step_id="acc_001", kind="accumulation"),
            SimpleNamespace(step_id="fin_09", kind="final_probe"),
            SimpleNamespace(step_id="pre_01", kind="pre_event_probe"),
        ]
    )


def test_rerun_probe_steps_takes_probe_ids_only():
    """A rerun throws a recorded run away; doing that to an accumulation step
    would leave the mainline world and its memory behind a ledger that says the
    step never ran."""
    plan = _rerun_plan()
    assert run_orchestrator._parse_rerun_probe_steps(None, plan) == []
    assert run_orchestrator._parse_rerun_probe_steps("fin_09, pre_01", plan) == [
        "fin_09",
        "pre_01",
    ]
    with pytest.raises(ValueError, match=r"\['acc_001'\]"):
        run_orchestrator._parse_rerun_probe_steps("fin_09,acc_001", plan)
    with pytest.raises(ValueError, match="invalid --rerun-probe-steps value"):
        run_orchestrator._parse_rerun_probe_steps(" , ", plan)


def test_superseding_a_probe_moves_its_step_dir_and_forgets_its_record(
    tmp_path: Path,
):
    run_dir = tmp_path / "run"
    ledger = RunLedger(run_dir, "run_001")
    for step_id in ("acc_001", "fin_09"):
        (run_dir / "steps" / step_id).mkdir(parents=True)
        (run_dir / "steps" / step_id / "transcript.json").write_text(
            "{}", encoding="utf-8"
        )
        ledger.mark_running(step_id)
        ledger.mark_done(step_id)
    (run_dir / "checkpoints" / "acc_001").mkdir(parents=True)

    moved = run_orchestrator._supersede_probe_steps(run_dir, ledger, ["fin_09"])

    assert [path.parent.name for path in moved] == ["steps_superseded"]
    assert moved[0].name.startswith("fin_09__")
    assert (moved[0] / "transcript.json").is_file()
    assert not (run_dir / "steps" / "fin_09").exists()
    assert ledger.decision_for("fin_09") == "run"
    # Nothing else moves: the mainline step, its ledger record, its checkpoint.
    assert (run_dir / "steps" / "acc_001" / "transcript.json").is_file()
    assert ledger.decision_for("acc_001") == "skip"
    assert (run_dir / "checkpoints" / "acc_001").is_dir()
    assert RunLedger(run_dir, "run_001").decision_for("fin_09") == "run"


def test_superseding_a_probe_goes_through_the_shared_run_layout_helper(
    tmp_path: Path, monkeypatch
):
    """One implementation of the layout, two callers of it.

    `--rerun-probe-steps` here and the in-batch void retry
    (`Orchestrator._run_one_probe`) must write the same
    `steps_superseded/<id>__<stamp>/`, because the cohort manifest and
    `--list-cohorts` read whichever of them wrote it. The ledger stays this
    caller's own business: the helper only moves the directory.
    """
    assert run_orchestrator.supersede_step_dirs is run_layout.supersede_step_dirs
    run_dir = tmp_path / "run"
    ledger = RunLedger(run_dir, "run_001")
    ledger.mark_done("fin_09")
    set_aside = run_dir / "steps_superseded" / "fin_09__20260916T010203Z"
    calls: list[tuple[Path, list[str]]] = []

    def spy(target_run_dir, step_ids):
        calls.append((Path(target_run_dir), list(step_ids)))
        return [set_aside]

    monkeypatch.setattr(run_orchestrator, "supersede_step_dirs", spy)

    moved = run_orchestrator._supersede_probe_steps(run_dir, ledger, ["fin_09"])

    assert calls == [(run_dir, ["fin_09"])]
    assert moved == [set_aside]
    assert ledger.decision_for("fin_09") == "run"


def test_setting_one_probe_aside_twice_in_a_second_writes_two_cohorts(
    tmp_path: Path,
):
    """The stamp is a cohort id, so one call may never reuse another's second.

    Three attempts (`harness.orchestrator.PROBE_ATTEMPTS`) set the same probe
    aside twice, and both set-asides normally happen inside the same second. On
    one shared name `shutil.move` does not fail — it moves the second
    directory *inside* the first, so the first attempt's evidence and the
    second attempt's directory end up tangled at a path the cohort manifest
    cannot read either of them back from.
    """
    run_dir = tmp_path / "run"
    cohorts = []
    for attempt in (1, 2):
        step_dir = run_dir / "steps" / "fin_01"
        step_dir.mkdir(parents=True)
        (step_dir / "transcript.json").write_text(
            f'{{"attempt": {attempt}}}', encoding="utf-8"
        )
        cohorts += run_layout.supersede_step_dirs(run_dir, ["fin_01"])

    assert len(cohorts) == 2
    assert len({path.name for path in cohorts}) == 2
    # Each cohort holds its own attempt's evidence, at the top level.
    for attempt, path in enumerate(cohorts, start=1):
        assert (path / "transcript.json").read_text(encoding="utf-8") == (
            f'{{"attempt": {attempt}}}'
        )
        assert not (path / "fin_01").exists()


def test_non_file_memory_ignores_global_consolidation_model():
    assert (
        run_orchestrator._effective_consolidation_model(
            "mem0",
            "globally-configured-model",
        )
        == ""
    )


def test_mem0_collection_name_is_stable_and_run_scoped():
    assert (
        run_orchestrator._mem0_collection_name("run/001")
        == "mempa_run_001_mem0"
    )
    assert (
        run_orchestrator._mem0_collection_name("run_002")
        != run_orchestrator._mem0_collection_name("run_001")
    )


def test_qdrant_endpoint_is_read_once_and_validated(monkeypatch):
    monkeypatch.setenv("QDRANT_HOST", "qdrant.internal")
    monkeypatch.setenv("QDRANT_PORT", "7444")

    assert run_orchestrator._qdrant_endpoint_from_environment("mem0") == (
        "qdrant.internal",
        7444,
    )
    assert run_orchestrator._qdrant_endpoint_from_environment("no_memory") == (
        "",
        0,
    )

    monkeypatch.setenv("QDRANT_PORT", "invalid")
    with pytest.raises(ValueError, match="value='invalid'"):
        run_orchestrator._qdrant_endpoint_from_environment("simple_rag")


def test_simple_rag_plan_freezes_qdrant_endpoint(authoring_root):
    mem0_runtime_config = _mem0_runtime_config()
    plan = run_plan_builder._build_frozen_run_plan(
        run_id="run_001",
        persona_id="user_a",
        memory_condition="simple_rag",
        pa_model="openai/gpt-5.5",
        script_names=["acc_001"],
        qdrant_host="qdrant.internal",
        qdrant_port=6333,
        mem0_runtime_config=mem0_runtime_config,
    )

    assert plan.qdrant_host == "qdrant.internal"
    assert plan.qdrant_port == 6333
    assert plan.mem0_runtime_config is not None
    assert plan.mem0_runtime_config.enable_entity_memory is False


def test_mem0_runtime_enables_entity_memory() -> None:
    config = run_orchestrator._effective_mem0_runtime_config("mem0")

    assert config is not None
    assert config.enable_entity_memory is True


@pytest.mark.parametrize(
    "constant_name",
    [
        "MEM0_EMBEDDING_MAX_INPUT_TOKENS",
        "MEM0_EMBEDDING_CHUNK_SAFETY_MARGIN_TOKENS",
        "MEM0_EXTRACTION_MAX_OUTPUT_TOKENS",
    ],
)
def test_mem0_chunk_policy_is_frozen_into_the_mem0_runtime_config(
    monkeypatch,
    tmp_path,
    constant_name: str,
) -> None:
    """Two runs that cut a pool differently freeze different mem0 configs.

    So do two runs whose extractor was allowed a different amount to say; that
    budget is `budgets.mem0_extraction_max_output_tokens` in the run config.
    """
    from memory.backends import mem0_adapter

    first = _mem0_runtime_config()

    if constant_name == "MEM0_EXTRACTION_MAX_OUTPUT_TOKENS":
        config = tmp_path / "run_config.yaml"
        config.write_text(yaml.safe_dump({"budgets": {
            "mem0_extraction_max_output_tokens": mem0_adapter.MEM0_EXTRACTION_MAX_OUTPUT_TOKENS - 1,
        }}))
        monkeypatch.setenv("MEMPA_RUN_CONFIG", str(config))
    else:
        monkeypatch.setattr(
            mem0_adapter,
            constant_name,
            getattr(mem0_adapter, constant_name) - 1,
        )
    assert _mem0_runtime_config() != first


def test_mem0_plan_freezes_the_chunk_tokenizer(authoring_root) -> None:
    plan = run_plan_builder._build_frozen_run_plan(
        run_id="run_001",
        persona_id="user_a",
        memory_condition="simple_rag",
        pa_model="openai/gpt-5.5",
        script_names=["acc_001"],
        qdrant_host="qdrant.internal",
        qdrant_port=6333,
        mem0_runtime_config=_mem0_runtime_config(),
    )

    assert plan.mem0_runtime_config is not None
    assert (
        plan.mem0_runtime_config.embedding_tokenizer_encoding
        == tiktoken.encoding_for_model(MEM0_OPENAI_EMBEDDING_MODEL).name
    )
    assert (
        plan.mem0_runtime_config.tiktoken_version
        == importlib.metadata.version("tiktoken")
    )
    assert plan.mem0_runtime_config.embedding_max_input_tokens == 8191
    assert plan.mem0_runtime_config.embedding_chunk_safety_margin_tokens == 256
    assert plan.mem0_runtime_config.extraction_max_output_tokens == 16000


def test_mem0_resume_can_keep_its_frozen_strategy_without_changing_other_fields():
    new = run_orchestrator._effective_mem0_runtime_config("mem0")
    old = run_orchestrator._effective_mem0_runtime_config(
        "mem0", extraction_chunk_strategy="legacy_no_message_split",
    )
    assert new.extraction_chunk_strategy == "message_split_v1"
    assert old.extraction_chunk_strategy == "legacy_no_message_split"
    assert old.model_dump(exclude={"extraction_chunk_strategy"}) == new.model_dump(
        exclude={"extraction_chunk_strategy"}
    )


# --- The strategy the *adapter* is built on -----------------------------------
#
# The helper above only says what the effective config computes. What decides
# how a live turn is cut is the value `Mem0Adapter` is constructed with, so
# these drive `_execute_benchmark_run` to the registry call and read the
# adapter's own keyword arguments -- for the mainline, and for the probe
# branch built from the same options.


class _StopAtSimulatorWorkspace(RuntimeError):
    pass


def _mem0_adapter_options_on_resume(
    tmp_path,
    monkeypatch,
    *,
    memory_condition: str,
    frozen_strategy: str | None,
    drop_strategy_key: bool = False,
    overrides: dict | None = None,
) -> list[tuple[str, dict]]:
    """Resume a frozen plan and hand back what the adapter was created with.

    `frozen_strategy=None` freezes the new-run default. `drop_strategy_key`
    then removes the field from the plan file, which is the shape of every
    plan frozen before 2026-09-18: absent, and read back as legacy.
    """
    model = "openai/gpt-5.5"
    api_base = "https://openrouter.ai/api/v1"
    mem0_config = run_orchestrator._effective_mem0_runtime_config(
        memory_condition,
        extraction_chunk_strategy=frozen_strategy,
    )
    assert mem0_config is not None
    if overrides:
        mem0_config = mem0_config.model_copy(update=overrides)
    plan = run_plan_builder._build_frozen_run_plan(
        run_id=f"{memory_condition}_strategy_001",
        persona_id="user_a",
        memory_condition=memory_condition,
        pa_model=model,
        script_names=["acc_001"],
        assistant_llm_provider_name="openrouter",
        assistant_llm_api_base=api_base,
        memory_side_model=model if memory_condition == "mem0" else "",
        qdrant_host="frozen-qdrant.internal",
        qdrant_port=7444,
        mem0_runtime_config=mem0_config,
    )
    plan_path = plan.dump_yaml(tmp_path / "run_plan.yaml")
    if drop_strategy_key:
        data = yaml.safe_load(plan_path.read_text(encoding="utf-8"))
        # The legacy value hashes as its absence, so the plan still verifies.
        del data["mem0_runtime_config"]["extraction_chunk_strategy"]
        plan_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
        assert "extraction_chunk_strategy" not in plan_path.read_text(
            encoding="utf-8"
        )

    monkeypatch.setattr(
        run_orchestrator,
        "_resolve_assistant_llm_provider_profile",
        lambda _: run_orchestrator.AssistantLLMProviderProfile(
            provider_name="openrouter", api_key="api-key", api_base=api_base,
        ),
    )
    monkeypatch.setenv("OPENAI_EMB_API_KEY", "embedding-key")

    class EmptyStateClient:
        def native_ids(self) -> set[str]:
            return set()

        def history_native_ids(self) -> set[str]:
            return set()

    monkeypatch.setattr(
        run_orchestrator, "create_mem0_client", lambda **_k: EmptyStateClient()
    )
    monkeypatch.setattr(
        run_orchestrator,
        "create_simple_rag_mem0_client",
        lambda **_k: EmptyStateClient(),
    )

    adapter_calls: list[tuple[str, dict]] = []

    class RecordingRegistry:
        def create(self, condition, **kwargs):
            adapter_calls.append((condition, kwargs))
            return SimpleNamespace()

    monkeypatch.setattr(
        run_orchestrator, "default_registry", lambda: RecordingRegistry()
    )
    monkeypatch.setattr(
        run_orchestrator,
        "_prepare_simulator_workspace",
        lambda **_kwargs: (_ for _ in ()).throw(_StopAtSimulatorWorkspace()),
    )
    args = SimpleNamespace(
        verbose=True,
        plan=str(plan_path),
        persona_id="user_a",
        memory_condition=memory_condition,
        pa_model=model,
        consolidation_model=None,
        memory_side_model=model if memory_condition == "mem0" else None,
        memos_api_base=None,
        assistant_llm_provider_name="openrouter",
        assistant_llm_api_base=api_base,
    )
    return adapter_calls, args


@pytest.mark.asyncio
async def test_a_plan_frozen_without_the_field_builds_a_legacy_adapter(
    tmp_path, monkeypatch, authoring_root
):
    """The 2026-09-19 shape: no field in the plan, legacy at the adapter.

    Flipping the new-run default must not reinterpret these runs. The
    assertion is on the adapter's own argument, not on the config helper: the
    helper agreeing while the adapter is built from the module constant is
    exactly the bug this covers.
    """
    adapter_calls, args = _mem0_adapter_options_on_resume(
        tmp_path,
        monkeypatch,
        memory_condition="mem0",
        frozen_strategy="legacy_no_message_split",
        drop_strategy_key=True,
    )

    with pytest.raises(_StopAtSimulatorWorkspace):
        await run_orchestrator._execute_benchmark_run(args)

    assert len(adapter_calls) == 1
    condition, options = adapter_calls[0]
    assert condition == "mem0"
    assert options["extraction_chunk_strategy"] == "legacy_no_message_split"


@pytest.mark.asyncio
async def test_a_plan_frozen_on_the_new_default_builds_a_splitting_adapter(
    tmp_path, monkeypatch, authoring_root
):
    """A plan frozen (or revised) on `message_split_v1` runs on it."""
    adapter_calls, args = _mem0_adapter_options_on_resume(
        tmp_path, monkeypatch, memory_condition="mem0", frozen_strategy=None,
    )

    with pytest.raises(_StopAtSimulatorWorkspace):
        await run_orchestrator._execute_benchmark_run(args)

    assert len(adapter_calls) == 1
    assert adapter_calls[0][1]["extraction_chunk_strategy"] == "message_split_v1"


@pytest.mark.asyncio
async def test_a_probe_branch_adapter_inherits_the_mainline_strategy(
    tmp_path, monkeypatch, authoring_root
):
    """A probe reads its branch under the mainline's strategy, not the default.

    The branch adapter is built from the mainline options, so the legacy run
    is the case that would break silently if the branch fell back to the
    module constant.
    """
    adapter_calls, args = _mem0_adapter_options_on_resume(
        tmp_path,
        monkeypatch,
        memory_condition="mem0",
        frozen_strategy="legacy_no_message_split",
        drop_strategy_key=True,
    )

    with pytest.raises(_StopAtSimulatorWorkspace):
        await run_orchestrator._execute_benchmark_run(args)

    mainline_options = adapter_calls[0][1]
    branch_dir = tmp_path / "branch"
    (branch_dir / "backend_state").mkdir(parents=True)
    await run_orchestrator._build_branch_memory_adapter(
        memory_condition="mem0",
        namespace="mempa_branch_mem0",
        branch_dir=branch_dir,
        mainline_options=mainline_options,
    )

    assert len(adapter_calls) == 2
    branch_options = adapter_calls[1][1]
    assert branch_options["extraction_chunk_strategy"] == "legacy_no_message_split"
    # Only the namespace moves: the branch client is its own, the rest of the
    # mainline options -- the strategy with them -- is carried over verbatim.
    assert {
        key: value for key, value in branch_options.items() if key != "client"
    } == {key: value for key, value in mainline_options.items() if key != "client"}


@pytest.mark.asyncio
async def test_a_revised_plan_preserves_its_frozen_extraction_budget(
    tmp_path, monkeypatch, authoring_root
):
    """A resume runs on the budget its plan carries, not today's default."""
    adapter_calls, args = _mem0_adapter_options_on_resume(
        tmp_path,
        monkeypatch,
        memory_condition="mem0",
        frozen_strategy="message_split_v1",
        overrides={"extraction_max_output_tokens": 32_000},
    )
    client_calls = []
    monkeypatch.setattr(
        run_orchestrator,
        "create_mem0_client",
        lambda **kwargs: (
            client_calls.append(kwargs)
            or SimpleNamespace(
                native_ids=lambda: set(), history_native_ids=lambda: set()
            )
        ),
    )

    with pytest.raises(_StopAtSimulatorWorkspace):
        await run_orchestrator._execute_benchmark_run(args)

    assert len(adapter_calls) == 1
    assert client_calls[0]["extraction_max_output_tokens"] == 32_000


@pytest.mark.asyncio
async def test_mem0_extraction_counts_into_the_runs_provider_avoidance(
    tmp_path, monkeypatch, authoring_root
):
    """The mainline mem0 client gets the run's state, the same object the PA and
    consolidation providers count into; probe branches get it through the same kwargs."""
    adapter_calls, args = _mem0_adapter_options_on_resume(
        tmp_path, monkeypatch, memory_condition="mem0", frozen_strategy=None,
    )
    client_calls = []
    monkeypatch.setattr(
        run_orchestrator,
        "create_mem0_client",
        lambda **kwargs: (
            client_calls.append(kwargs)
            or SimpleNamespace(
                native_ids=lambda: set(), history_native_ids=lambda: set()
            )
        ),
    )

    with pytest.raises(_StopAtSimulatorWorkspace):
        await run_orchestrator._execute_benchmark_run(args)

    avoidance = client_calls[0]["provider_avoidance"]
    assert avoidance.path == tmp_path / "provider_avoidance.json"
    assert avoidance.threshold == 3


@pytest.mark.asyncio
async def test_a_simple_rag_plan_frozen_without_the_field_still_resumes(
    tmp_path, monkeypatch, authoring_root
):
    """The `infer=False` path is unchanged, and its old plans must load.

    simple_rag does its own raw-record splitting and never reads the strategy,
    but it shares `Mem0RuntimeConfig` -- so the new default would have failed
    its effective-config check if the resume did not select the frozen value.
    """
    adapter_calls, args = _mem0_adapter_options_on_resume(
        tmp_path,
        monkeypatch,
        memory_condition="simple_rag",
        frozen_strategy="legacy_no_message_split",
        drop_strategy_key=True,
    )

    with pytest.raises(_StopAtSimulatorWorkspace):
        await run_orchestrator._execute_benchmark_run(args)

    assert len(adapter_calls) == 1
    condition, options = adapter_calls[0]
    assert condition == "simple_rag"
    # Unchanged behaviour: simple_rag is not handed a strategy at all.
    assert "extraction_chunk_strategy" not in options


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("memory_condition", "expected_factory"),
    [("mem0", "openrouter"), ("simple_rag", "simple_rag")],
)
async def test_runner_selects_memory_factory_and_passes_frozen_plan_values(
    tmp_path,
    monkeypatch,
    authoring_root,
    memory_condition,
    expected_factory,
):
    model = "openai/gpt-5.5"
    api_base = "https://openrouter.ai/api/v1"
    plan = run_plan_builder._build_frozen_run_plan(
        run_id=f"{memory_condition}_run_001",
        persona_id="user_a",
        memory_condition=memory_condition,
        pa_model=model,
        script_names=["acc_001"],
        assistant_llm_provider_name="openrouter",
        assistant_llm_api_base=api_base,
        memory_side_model=model if memory_condition == "mem0" else "",
        qdrant_host="frozen-qdrant.internal",
        qdrant_port=7444,
        mem0_runtime_config=run_orchestrator._effective_mem0_runtime_config(
            memory_condition
        ),
    )
    plan_path = plan.dump_yaml(tmp_path / "run_plan.yaml")
    profile = run_orchestrator.AssistantLLMProviderProfile(
        provider_name="openrouter",
        api_key="api-key",
        api_base=api_base,
    )
    monkeypatch.setattr(
        run_orchestrator,
        "_resolve_assistant_llm_provider_profile",
        lambda _: profile,
    )
    monkeypatch.setenv("QDRANT_HOST", "changed-after-plan.example")
    monkeypatch.setenv("QDRANT_PORT", "9999")
    monkeypatch.setenv("OPENAI_EMB_API_KEY", "embedding-key")

    calls: dict[str, list[dict]] = {"openrouter": [], "simple_rag": []}

    class EmptyStateClient:
        def native_ids(self) -> set[str]:
            return set()

        def history_native_ids(self) -> set[str]:
            return set()

    def openrouter_factory(**kwargs):
        calls["openrouter"].append(kwargs)
        return EmptyStateClient()

    def simple_rag_factory(**kwargs):
        calls["simple_rag"].append(kwargs)
        return EmptyStateClient()

    monkeypatch.setattr(
        run_orchestrator,
        "create_mem0_client",
        openrouter_factory,
    )
    monkeypatch.setattr(
        run_orchestrator,
        "create_simple_rag_mem0_client",
        simple_rag_factory,
    )

    class StopAfterFactory(RuntimeError):
        pass

    def stop_after_factory(*_args, **_kwargs):
        raise StopAfterFactory

    # Persona loading now starts with the workspace copy; stop there.
    monkeypatch.setattr(
        run_orchestrator, "_prepare_simulator_workspace", stop_after_factory
    )
    args = SimpleNamespace(
        verbose=True,
        plan=str(plan_path),
        persona_id="user_a",
        memory_condition=memory_condition,
        pa_model=model,
        consolidation_model=None,
        memory_side_model=model if memory_condition == "mem0" else None,
        memos_api_base=None,
        assistant_llm_provider_name="openrouter",
        assistant_llm_api_base=api_base,
    )

    with pytest.raises(StopAfterFactory):
        await run_orchestrator._execute_benchmark_run(args)

    assert len(calls[expected_factory]) == 1
    unused_factory = "simple_rag" if expected_factory == "openrouter" else "openrouter"
    assert calls[unused_factory] == []
    factory_kwargs = calls[expected_factory][0]
    assert factory_kwargs["collection_name"] == (
        f"mempa_{memory_condition}_run_001_mem0"
    )
    assert factory_kwargs["qdrant_host"] == "frozen-qdrant.internal"
    assert factory_kwargs["qdrant_port"] == 7444
    assert factory_kwargs["expected_run_id"] == f"{memory_condition}_run_001"
    assert factory_kwargs["allow_existing_collection"] is True
    assert factory_kwargs["embedding_api_base"] == "https://api.openai.com/v1"
    assert factory_kwargs["embedding_model"] == "text-embedding-3-small"
    assert factory_kwargs["embedding_dimensions"] == 1536
    assert factory_kwargs["qdrant_distance"] == "Cosine"
    assert factory_kwargs["qdrant_on_disk"] is True
    assert factory_kwargs["qdrant_bm25_enabled"] is False
    assert factory_kwargs["enable_entity_memory"] is (
        memory_condition == "mem0"
    )
    assert factory_kwargs["history_db_path"] == (
        tmp_path / "backend_state" / "mem0_history.sqlite3"
    )
    assert factory_kwargs["allow_existing_history_db"] is True
    assert factory_kwargs["require_existing_history_db"] is False
    if expected_factory == "openrouter":
        assert factory_kwargs["memory_side_model"] == model
        assert factory_kwargs["provider_name"] == "openrouter"
        assert factory_kwargs["api_key"] == "api-key"


@pytest.mark.asyncio
async def test_runner_rejects_unsafe_resume_before_creating_memory_client(
    tmp_path,
    monkeypatch,
    authoring_root,
):
    monkeypatch.setenv("OPENAI_EMB_API_KEY", "embedding-key")
    model = "openai/gpt-5.5"
    plan = run_plan_builder._build_frozen_run_plan(
        run_id="failed_run_001",
        persona_id="user_a",
        memory_condition="simple_rag",
        pa_model=model,
        script_names=["acc_001"],
        qdrant_host="frozen-qdrant.internal",
        qdrant_port=6333,
        mem0_runtime_config=_mem0_runtime_config(),
    )
    plan_path = plan.dump_yaml(tmp_path / "run_plan.yaml")
    ledger = RunLedger(tmp_path, plan.run_id)
    ledger.mark_failed("acc_001", "partial external side effects")
    profile = run_orchestrator.AssistantLLMProviderProfile(
        provider_name="openrouter",
        api_key="test-key",
    )
    monkeypatch.setattr(
        run_orchestrator,
        "_resolve_assistant_llm_provider_profile",
        lambda _: profile,
    )

    def unexpected_client_creation(**_kwargs):
        raise AssertionError("memory client must not be created")

    monkeypatch.setattr(
        run_orchestrator,
        "create_simple_rag_mem0_client",
        unexpected_client_creation,
    )
    args = SimpleNamespace(
        verbose=True,
        plan=str(plan_path),
        persona_id="user_a",
        memory_condition="simple_rag",
        pa_model=model,
        consolidation_model=None,
        memory_side_model=None,
        memos_api_base=None,
        assistant_llm_provider_name="openrouter",
        assistant_llm_api_base=None,
    )

    with pytest.raises(UnsafeRunResumeError, match="Start a clean run"):
        await run_orchestrator._execute_benchmark_run(args)


class _RollbackReached(RuntimeError):
    """Sentinel: the rollback ran, so the test need not run the benchmark."""


@pytest.mark.asyncio
@pytest.mark.parametrize("memory_condition", ["simple_rag", "mem0", "memos"])
async def test_rollback_of_an_external_backend_run_has_its_client(
    memory_condition,
    tmp_path,
    monkeypatch,
    authoring_root,
):
    """A run stopped mid-step on an external backend rolls back like a local one.

    `_execute_benchmark_run` used to call `resume_or_restore` before it built
    the memory client, so `CheckpointManager.backend_client` was still None
    when the rollback reached `_restore_backend_state`: the restore refused
    with `IncompleteCheckpointError` and the run could not be resumed at all.
    Only `nanobot_file_memory`, whose state lives inside the run directory,
    came back. Every condition whose memory lives outside it is covered here,
    because binding one client early does not bind the others: the MemOS
    client reached `memory_adapter_options` a second time, after the rollback,
    and went on failing once mem0 and simple_rag were fixed.
    """
    monkeypatch.setenv("OPENAI_EMB_API_KEY", "embedding-key")
    model = "openai/gpt-5.5"
    created: list = []
    plan_kwargs = dict(
        run_id="rollback_run_001",
        persona_id="user_a",
        memory_condition=memory_condition,
        pa_model=model,
        script_names=["acc_001"],
    )
    memos_api_base = "http://127.0.0.1:8100"

    if memory_condition == "memos":
        memos_config = run_orchestrator._effective_memos_runtime_config(
            memory_condition="memos",
            api_base=memos_api_base,
            memory_side_provider_name="openrouter",
            memory_side_model=model,
            memory_side_api_base="https://openrouter.ai/api/v1",
        )
        plan_kwargs.update(
            memory_side_model=model,
            memos_runtime_config=memos_config,
        )

        class FakeMemOSHttpClient:
            def __init__(self, **_kwargs):
                created.append(self)

            def health(self):
                # A no-op: the rollback has to be reachable without a live
                # MemOS service answering.
                return {}

        monkeypatch.setattr(
            run_orchestrator, "MemOSHttpClient", FakeMemOSHttpClient
        )
    else:
        config = run_orchestrator._effective_mem0_runtime_config(memory_condition)
        assert config is not None
        plan_kwargs.update(
            qdrant_host="frozen-qdrant.internal",
            qdrant_port=6333,
            mem0_runtime_config=config,
        )
        if memory_condition == "mem0":
            plan_kwargs.update(memory_side_model=model)

        def fake_client_factory(**_kwargs):
            client = object()
            created.append(client)
            return client

        monkeypatch.setattr(
            run_orchestrator,
            (
                "create_mem0_client"
                if memory_condition == "mem0"
                else "create_simple_rag_mem0_client"
            ),
            fake_client_factory,
        )

    plan = run_plan_builder._build_frozen_run_plan(**plan_kwargs)
    plan_path = plan.dump_yaml(tmp_path / "run_plan.yaml")
    ledger = RunLedger(tmp_path, plan.run_id)
    ledger.mark_failed("acc_001", "died mid-step")
    profile = run_orchestrator.AssistantLLMProviderProfile(
        provider_name="openrouter",
        api_key="test-key",
    )
    monkeypatch.setattr(
        run_orchestrator,
        "_resolve_assistant_llm_provider_profile",
        lambda _: profile,
    )
    monkeypatch.setattr(
        run_orchestrator.CheckpointManager,
        "list_checkpoints",
        lambda self: ["acc_001"],
    )
    monkeypatch.setattr(
        run_orchestrator.CheckpointManager,
        "metadata",
        lambda self, checkpoint_id: {"timestamp": "20260913T000000"},
    )

    restored_with: list = []

    def fake_restore(self, checkpoint_id, *, ledger):
        # The real restore reaches the external backend through exactly this
        # accessor, and raises IncompleteCheckpointError when nothing is bound.
        restored_with.append((checkpoint_id, self._require_backend_client()))
        raise _RollbackReached(checkpoint_id)

    monkeypatch.setattr(
        run_orchestrator.CheckpointManager, "restore", fake_restore
    )

    args = SimpleNamespace(
        verbose=True,
        plan=str(plan_path),
        persona_id="user_a",
        memory_condition=memory_condition,
        pa_model=model,
        consolidation_model=None,
        memory_side_model=None if memory_condition == "simple_rag" else model,
        memos_api_base=memos_api_base if memory_condition == "memos" else None,
        assistant_llm_provider_name="openrouter",
        assistant_llm_api_base=None,
    )

    with pytest.raises(_RollbackReached, match="acc_001"):
        await run_orchestrator._execute_benchmark_run(args)

    assert len(created) == 1
    assert restored_with == [("acc_001", created[0])]


def test_new_memos_plans_freeze_distinct_cube_ids(authoring_root):
    model = "openai/gpt-5.5"
    provider_api_base = "https://openrouter.ai/api/v1"
    memos_config = run_orchestrator._effective_memos_runtime_config(
        memory_condition="memos",
        api_base="http://127.0.0.1:8100",
        memory_side_provider_name="openrouter",
        memory_side_model=model,
        memory_side_api_base=provider_api_base,
    )
    assert memos_config is not None
    kwargs = {
        "run_id": "memos_run_001",
        "persona_id": "user_a",
        "memory_condition": "memos",
        "pa_model": model,
        "script_names": ["acc_001"],
        "assistant_llm_provider_name": "openrouter",
        "assistant_llm_api_base": provider_api_base,
        "memory_side_model": model,
        "memos_runtime_config": memos_config,
    }

    first = run_plan_builder._build_frozen_run_plan(**kwargs)
    second = run_plan_builder._build_frozen_run_plan(**kwargs)

    assert first.memos_cube_id.startswith("mempa_memos_run_001_memos_")
    assert first.memos_cube_id != second.memos_cube_id


class _FakeMemOSStateClient:
    def __init__(self, grouped: dict) -> None:
        self.grouped = grouped
        self.get_requests: list[dict] = []

    def get_memories(self, request: dict) -> dict:
        self.get_requests.append(request)
        return self.grouped


class _FakeMem0StateClient:
    def __init__(
        self,
        *,
        active_ids: list[str],
        history_ids: list[str],
    ) -> None:
        self._active_ids = set(active_ids)
        self._history_ids = set(history_ids)

    def native_ids(self) -> set[str]:
        return set(self._active_ids)

    def history_native_ids(self) -> set[str]:
        return set(self._history_ids)


def test_mem0_backend_state_accepts_exact_committed_ids(tmp_path):
    plan = SimpleNamespace(run_id="run_001", memory_condition="simple_rag")
    run_ledger = RunLedger(tmp_path, plan.run_id)
    run_ledger.mark_done("acc_001")
    recorder = MemoryIORecorder(tmp_path / "memory_io.jsonl", run_id=plan.run_id)
    _record_committed_mem0_ids(recorder, ["memory_001"])
    client = _FakeMem0StateClient(
        active_ids=["memory_001"],
        history_ids=["memory_001"],
    )

    run_orchestrator._validate_mem0_backend_state(
        client=client,
        plan=plan,
        memory_io_recorder=recorder,
        run_ledger=run_ledger,
    )


@pytest.mark.parametrize(
    ("active_ids", "history_ids", "error_fragment"),
    [
        ([], ["memory_001"], "missing_active_ids=['memory_001']"),
        (
            ["memory_001", "memory_002"],
            ["memory_001"],
            "extra_active_ids=['memory_002']",
        ),
        (["memory_001"], [], "missing_history_ids=['memory_001']"),
        (
            ["memory_001"],
            ["memory_001", "memory_002"],
            "extra_history_ids=['memory_002']",
        ),
    ],
)
def test_mem0_backend_state_rejects_missing_or_extra_ids(
    tmp_path,
    active_ids,
    history_ids,
    error_fragment,
):
    plan = SimpleNamespace(run_id="run_001", memory_condition="simple_rag")
    run_ledger = RunLedger(tmp_path, plan.run_id)
    run_ledger.mark_done("acc_001")
    recorder = MemoryIORecorder(tmp_path / "memory_io.jsonl", run_id=plan.run_id)
    _record_committed_mem0_ids(recorder, ["memory_001"])

    with pytest.raises(RuntimeError, match="does not match") as exc_info:
        run_orchestrator._validate_mem0_backend_state(
            client=_FakeMem0StateClient(
                active_ids=active_ids,
                history_ids=history_ids,
            ),
            plan=plan,
            memory_io_recorder=recorder,
            run_ledger=run_ledger,
        )

    assert error_fragment in str(exc_info.value)


def test_mem0_resume_rejects_recorded_steps_without_memory_audit(tmp_path):
    plan = SimpleNamespace(run_id="run_001", memory_condition="simple_rag")
    run_ledger = RunLedger(tmp_path, plan.run_id)
    run_ledger.mark_done("acc_001")
    recorder = MemoryIORecorder(tmp_path / "memory_io.jsonl", run_id=plan.run_id)

    with pytest.raises(RuntimeError, match="recorded steps but no memory audit"):
        run_orchestrator._validate_mem0_backend_state(
            client=_FakeMem0StateClient(active_ids=[], history_ids=[]),
            plan=plan,
            memory_io_recorder=recorder,
            run_ledger=run_ledger,
        )


def test_mem0_rejects_committed_audit_without_ledger_steps(tmp_path):
    plan = SimpleNamespace(run_id="run_001", memory_condition="simple_rag")
    recorder = MemoryIORecorder(tmp_path / "memory_io.jsonl", run_id=plan.run_id)
    _record_committed_mem0_ids(recorder, ["memory_001"])

    with pytest.raises(RuntimeError, match="audit records but no ledger steps"):
        run_orchestrator._validate_mem0_backend_state(
            client=_FakeMem0StateClient(
                active_ids=["memory_001"],
                history_ids=["memory_001"],
            ),
            plan=plan,
            memory_io_recorder=recorder,
            run_ledger=RunLedger(tmp_path, plan.run_id),
        )


def test_memos_backend_state_accepts_exact_committed_ids(tmp_path):
    plan = SimpleNamespace(
        run_id="run_001",
        persona_id="user_a",
        memos_cube_id="cube_001",
    )
    recorder = MemoryIORecorder(tmp_path / "memory_io.jsonl", run_id=plan.run_id)
    _record_committed_memos_ids(recorder, ["memory_001"])
    client = _FakeMemOSStateClient(
        _memos_grouped(plan.memos_cube_id, ["memory_001"])
    )

    run_orchestrator._validate_memos_backend_state(
        client=client,
        plan=plan,
        memory_io_recorder=recorder,
        run_ledger=RunLedger(tmp_path, plan.run_id),
    )

    assert client.get_requests == [
        {
            "mem_cube_id": "cube_001",
            "user_id": "user_a",
            "include_preference": False,
            "include_tool_memory": False,
            "include_skill_memory": False,
        }
    ]


@pytest.mark.parametrize(
    ("committed_ids", "stored_ids", "error_fragment"),
    [
        (["memory_001"], [], "missing_ids=['memory_001']"),
        ([], ["memory_002"], "extra_ids=['memory_002']"),
    ],
)
def test_memos_backend_state_rejects_missing_or_extra_ids(
    tmp_path,
    committed_ids,
    stored_ids,
    error_fragment,
):
    plan = SimpleNamespace(
        run_id="run_001",
        persona_id="user_a",
        memos_cube_id="cube_001",
    )
    recorder = MemoryIORecorder(tmp_path / "memory_io.jsonl", run_id=plan.run_id)
    if committed_ids:
        _record_committed_memos_ids(recorder, committed_ids)
    client = _FakeMemOSStateClient(
        _memos_grouped(plan.memos_cube_id, stored_ids)
    )

    with pytest.raises(RuntimeError, match="does not match") as exc_info:
        run_orchestrator._validate_memos_backend_state(
            client=client,
            plan=plan,
            memory_io_recorder=recorder,
            run_ledger=RunLedger(tmp_path, plan.run_id),
        )

    assert error_fragment in str(exc_info.value)


def test_memos_resume_rejects_recorded_steps_without_memory_audit(tmp_path):
    plan = SimpleNamespace(
        run_id="run_001",
        persona_id="user_a",
        memos_cube_id="cube_001",
    )
    run_ledger = RunLedger(tmp_path, plan.run_id)
    run_ledger.mark_done("acc_001")
    recorder = MemoryIORecorder(tmp_path / "memory_io.jsonl", run_id=plan.run_id)
    client = _FakeMemOSStateClient(_memos_grouped(plan.memos_cube_id, []))

    with pytest.raises(RuntimeError, match="recorded steps but no memory audit"):
        run_orchestrator._validate_memos_backend_state(
            client=client,
            plan=plan,
            memory_io_recorder=recorder,
            run_ledger=run_ledger,
        )


@pytest.mark.asyncio
async def test_runner_injects_frozen_memos_client_and_adapter_configuration(
    tmp_path,
    monkeypatch,
    authoring_root,
):
    monkeypatch.setenv("OPENAI_EMB_API_KEY", "embedding-key")
    model = "openai/gpt-5.5"
    provider_api_base = "https://openrouter.ai/api/v1"
    memos_api_base = "http://127.0.0.1:8100"
    memos_config = run_orchestrator._effective_memos_runtime_config(
        memory_condition="memos",
        api_base=memos_api_base,
        memory_side_provider_name="openrouter",
        memory_side_model=model,
        memory_side_api_base=provider_api_base,
    )
    assert memos_config is not None
    plan = run_plan_builder._build_frozen_run_plan(
        run_id="memos_run_001",
        persona_id="user_a",
        memory_condition="memos",
        pa_model=model,
        script_names=["acc_001"],
        assistant_llm_provider_name="openrouter",
        assistant_llm_api_base=provider_api_base,
        memory_side_model=model,
        memos_runtime_config=memos_config,
    )
    plan_path = plan.dump_yaml(tmp_path / "run_plan.yaml")
    profile = run_orchestrator.AssistantLLMProviderProfile(
        provider_name="openrouter",
        api_key="api-id:api-key",
        api_base=provider_api_base,
    )
    monkeypatch.setattr(
        run_orchestrator,
        "_resolve_assistant_llm_provider_profile",
        lambda _: profile,
    )

    client_init_calls: list[dict] = []

    class FakeMemOSHttpClient:
        def __init__(self, **kwargs):
            client_init_calls.append(kwargs)

        def health(self):
            return {
                "status": "healthy",
                "service": "memos",
                "version": "2.0.15",
                "mempa_patch_version": MEMOS_MEMPA_PATCH_VERSION,
            }

        def get_memories(self, request):
            return _memos_grouped(plan.memos_cube_id, [])

    adapter_calls: list[tuple[str, dict]] = []

    class FakeRegistry:
        def create(self, condition, **kwargs):
            adapter_calls.append((condition, kwargs))
            return SimpleNamespace()

    monkeypatch.setattr(run_orchestrator, "MemOSHttpClient", FakeMemOSHttpClient)
    monkeypatch.setattr(run_orchestrator, "default_registry", lambda: FakeRegistry())

    class StopAfterFactory(RuntimeError):
        pass

    def stop_after_factory(*_args, **_kwargs):
        raise StopAfterFactory

    # Persona loading now starts with the workspace copy; stop there.
    monkeypatch.setattr(
        run_orchestrator, "_prepare_simulator_workspace", stop_after_factory
    )
    args = SimpleNamespace(
        verbose=True,
        plan=str(plan_path),
        persona_id="user_a",
        memory_condition="memos",
        pa_model=model,
        consolidation_model=None,
        memory_side_model=model,
        memos_api_base=memos_api_base,
        assistant_llm_provider_name="openrouter",
        assistant_llm_api_base=provider_api_base,
    )

    with pytest.raises(StopAfterFactory):
        await run_orchestrator._execute_benchmark_run(args)

    assert client_init_calls == [
        {
            "api_base": memos_api_base,
            "timeout_seconds": 180.0,
        }
    ]
    assert len(adapter_calls) == 1
    condition, kwargs = adapter_calls[0]
    assert condition == "memos"
    assert kwargs["cube_id"] == plan.memos_cube_id
    assert kwargs["memory_session_id"] == "memos_run_001_main"
    assert kwargs["add_mode"] == "fine"
    assert kwargs["search_mode"] == "fast"
    assert kwargs["relativity"] == 0.45


def test_mem0_requires_memory_side_model_before_plan_creation():
    with pytest.raises(ValueError, match="mem0 requires memory_side_model"):
        run_plan_builder._build_frozen_run_plan(
            run_id="run_001",
            persona_id="user_a",
            memory_condition="mem0",
            pa_model="openai/gpt-5.5",
            script_names=["acc_001"],
            assistant_llm_provider_name="openrouter",
        )


def test_non_file_plan_rejects_consolidation_configuration():
    with pytest.raises(
        ValueError,
        match="only valid for nanobot_file_memory",
    ):
        run_plan_builder._build_frozen_run_plan(
            run_id="run_001",
            persona_id="user_a",
            memory_condition="mem0",
            pa_model="openai/gpt-5.5",
            script_names=["acc_001"],
            consolidation_model="irrelevant-model",
        )


def test_formal_file_memory_plan_rejects_missing_consolidation_model():
    with pytest.raises(
        ValueError,
        match="nanobot_file_memory requires consolidation_model",
    ):
        run_plan_builder._build_frozen_run_plan(
            run_id="run_001",
            persona_id="user_a",
            memory_condition="nanobot_file_memory",
            pa_model="openai/gpt-5.5",
            script_names=["acc_001"],
        )


def test_build_plan_sets_main_pa_session_key(authoring_root):
    plan = run_plan_builder._build_frozen_run_plan(
        run_id="user_a_NanoBotFileMemory_openai_gpt55_cfgaaaaaaaaaa_r01",
        persona_id="user_a",
        memory_condition="nanobot_file_memory",
        pa_model="openai/gpt-5.5",
        script_names=["acc_001"],
        consolidation_model="fixed-memory-model",
    )

    assert (
        plan.pa_session_key
        == "user_a_NanoBotFileMemory_openai_gpt55_cfgaaaaaaaaaa_r01_main"
    )
    assert plan.consolidation_model == "fixed-memory-model"
    assert plan.assistant_llm_provider_name == "openrouter"
    assert (
        plan.assistant_llm_api_base
        == run_plan_builder.OPENROUTER_API_BASE
    )
    assert plan.pa_request_policy["model"] == "openai/gpt-5.5"
    assert plan.consolidation_request_policy is not None
    assert [plan.pa_session_key_for_step(step) for step in plan.steps] == [
        "user_a_NanoBotFileMemory_openai_gpt55_cfgaaaaaaaaaa_r01_main",
    ]


def test_build_plan_uses_script_meta_for_target_labels(authoring_root):
    plan = run_plan_builder._build_frozen_run_plan(
        run_id="user_a_NanoBotFileMemory_openai_gpt55_cfgaaaaaaaaaa_r01",
        persona_id="user_a",
        memory_condition="nanobot_file_memory",
        pa_model="openai/gpt-5.5",
        script_names=["acc_001"],
        consolidation_model="fixed-memory-model",
    )

    step = plan.steps[0]
    assert step.context == "work"
    assert step.target_cell == "work_verbosity"
    assert step.target_attribute == "verbosity"
    assert step.target_setting == "Detailed"
    assert step.session_type == "stable"
    assert step.domain == "literature_and_citations"
    assert step.story_id == "summarize_anyons_review_for_draft"
    assert step.timeline_slot == "cycle1_work"
    assert step.time_position == "early_oct"
    assert not hasattr(step, "probe_polarity")
    assert step.benchmark_session_timestamp.isoformat() == (
        "2026-10-01T09:00:00+00:00"
    )


def test_build_plan_freezes_increasing_benchmark_session_timestamps(
    authoring_root,
):
    plan = run_plan_builder._build_frozen_run_plan(
        run_id="timeline_run_001",
        persona_id="user_a",
        memory_condition="no_memory",
        pa_model="openai/gpt-5.5",
        script_names=["acc_001", "acc_075"],
    )

    timestamps = [step.benchmark_session_timestamp for step in plan.steps]
    assert timestamps == sorted(timestamps)
    assert timestamps[1] > timestamps[0]
    assert plan.steps[1].kind == "forced_event"


def test_script_path_does_not_fall_back_to_noncanonical_locations(
    tmp_path,
    monkeypatch,
):
    noncanonical_path = tmp_path / "data/scripts/user_a/acc_001.yaml"
    noncanonical_path.parent.mkdir(parents=True)
    noncanonical_path.write_text("session_id: noncanonical\n", encoding="utf-8")
    monkeypatch.setattr(run_plan_builder, "PROJECT_ROOT", tmp_path)

    expected_path = tmp_path / "Authoring/sessions/normal/drafts/user_a/acc_001.yaml"
    with pytest.raises(FileNotFoundError, match=str(expected_path)):
        run_plan_builder._script_path(
            "user_a",
            "acc_001",
            outline_entry={"session_type": "stable"},
        )


def test_forced_transcript_path_uses_acc_num_not_persona_slot(tmp_path, monkeypatch):
    monkeypatch.setattr(run_plan_builder, "PROJECT_ROOT", tmp_path)
    path = (
        tmp_path
        / "Authoring/sessions/forced/transcripts/user_b/evolv_04/transcript.yaml"
    )
    path.parent.mkdir(parents=True)
    path.write_text("event: {}\n", encoding="utf-8")
    assert run_plan_builder._script_path(
        "user_b",
        "ignored",
        outline_entry={
            "session_type": "evolving_event",
            "acc_num": 60,
            "timeline_slot": "personal_topic_management",
        },
    ) == path


def test_incomplete_authoring_outline_is_rejected(tmp_path):
    path = tmp_path / "outline.yaml"
    path.write_text(
        "meta:\n  status: in_progress\nsessions: []\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="not complete"):
        run_plan_builder._load_outline_sessions_by_acc_num(path)


def _write_in_progress_outline(path, pending):
    path.write_text(
        yaml.safe_dump(
            {
                "meta": {"status": "in_progress", "pending_acc_nums": pending},
                "sessions": [
                    {"identity": {"acc_num": 1}, "story": {}},
                    {"identity": {"acc_num": 2}, "story": {}},
                ],
            }
        ),
        encoding="utf-8",
    )


def test_in_progress_outline_loads_when_requested_sessions_are_live(tmp_path):
    path = tmp_path / "outline.yaml"
    _write_in_progress_outline(path, [5, 7])

    sessions = run_plan_builder._load_outline_sessions_by_acc_num(
        path,
        requested_acc_nums=[1, 2],
    )

    assert sorted(sessions) == [1, 2]


def test_in_progress_outline_rejects_a_pending_requested_session(tmp_path):
    path = tmp_path / "outline.yaml"
    _write_in_progress_outline(path, [2, 5])

    with pytest.raises(ValueError) as exc_info:
        run_plan_builder._load_outline_sessions_by_acc_num(
            path,
            requested_acc_nums=[1, 2],
        )

    message = str(exc_info.value)
    assert str(path) in message
    assert "pending_acc_nums=[2]" in message


def test_in_progress_outline_rejects_a_requested_session_that_is_absent(tmp_path):
    path = tmp_path / "outline.yaml"
    _write_in_progress_outline(path, [5])

    with pytest.raises(ValueError) as exc_info:
        run_plan_builder._load_outline_sessions_by_acc_num(
            path,
            requested_acc_nums=[1, 9],
        )

    message = str(exc_info.value)
    assert str(path) in message
    assert "missing_acc_nums=[9]" in message


def test_in_progress_outline_without_requested_sessions_is_rejected(tmp_path):
    path = tmp_path / "outline.yaml"
    _write_in_progress_outline(path, [5])

    with pytest.raises(ValueError, match="not complete"):
        run_plan_builder._load_outline_sessions_by_acc_num(path)


def test_in_progress_outline_without_a_pending_list_is_rejected(tmp_path):
    path = tmp_path / "outline.yaml"
    _write_in_progress_outline(path, None)

    with pytest.raises(ValueError, match="meta.pending_acc_nums"):
        run_plan_builder._load_outline_sessions_by_acc_num(
            path,
            requested_acc_nums=[1],
        )


@pytest.mark.parametrize(
    ("outline", "message_fragment"),
    [
        (None, "root_type=NoneType value=None"),
        (["acc_001"], "root_type=list value=['acc_001']"),
        ({"sessions": {}}, "'sessions' must be a list"),
        ({"sessions": ["acc_001"]}, "sessions_index=0 value='acc_001'"),
        (
            {"sessions": [{"story": {"story_id": "missing_identity"}}]},
            "requires identity and story groups",
        ),
        (
            {"sessions": [{"identity": {"acc_num": 1}}]},
            "requires identity and story groups",
        ),
        (
            {"sessions": [{"identity": {"session_type": "stable"}, "story": {}}]},
            "requires identity.acc_num",
        ),
        (
            {"sessions": [{"identity": {"acc_num": "1"}, "story": {}}]},
            "value='1'",
        ),
        (
            {"sessions": [{"identity": {"acc_num": 1.5}, "story": {}}]},
            "value=1.5",
        ),
        (
            {"sessions": [{"identity": {"acc_num": 0}, "story": {}}]},
            "value=0",
        ),
        (
            {
                "sessions": [
                    {"identity": {"acc_num": 1}, "story": {}},
                    {"identity": {"acc_num": 1}, "story": {}},
                ]
            },
            "Duplicate session outline acc_num",
        ),
    ],
)
def test_outline_loader_rejects_invalid_existing_files(
    tmp_path,
    outline,
    message_fragment,
):
    path = tmp_path / "outline.yaml"
    path.write_text(yaml.safe_dump(outline), encoding="utf-8")

    with pytest.raises(ValueError) as exc_info:
        run_plan_builder._load_outline_sessions_by_acc_num(path)

    message = str(exc_info.value)
    assert str(path) in message
    assert message_fragment in message


@pytest.mark.asyncio
async def test_loop_factory_uses_injected_workspace_adapter_and_recorder(
    tmp_path,
    monkeypatch,
    authoring_root,
):
    model = "openai/gpt-5.5"
    plan = run_plan_builder._build_frozen_run_plan(
        run_id="loopctx_run_001",
        persona_id="user_a",
        memory_condition="no_memory",
        pa_model=model,
        script_names=["acc_001"],
    )
    plan_path = plan.dump_yaml(tmp_path / "run_plan.yaml")
    profile = run_orchestrator.AssistantLLMProviderProfile(
        provider_name="openrouter",
        api_key="api-key",
    )
    monkeypatch.setattr(
        run_orchestrator,
        "_resolve_assistant_llm_provider_profile",
        lambda _: profile,
    )

    fixture = authoring_root / "data" / "state_fixtures" / "user_a"
    fixture.mkdir(parents=True, exist_ok=True)
    (fixture / "base.txt").write_text("base", encoding="utf-8")

    captured: dict = {}

    class CapturingSessionExecutor:
        def __init__(self, loop_factory):
            captured["factory"] = loop_factory

    class StopBeforeRun(RuntimeError):
        pass

    def stop(**_kwargs):
        raise StopBeforeRun

    monkeypatch.setattr(
        run_orchestrator, "SessionExecutor", CapturingSessionExecutor
    )
    monkeypatch.setattr(run_orchestrator, "Orchestrator", stop)
    monkeypatch.setattr(
        run_orchestrator,
        "PersonaIdentity",
        SimpleNamespace(from_yaml=lambda _path: object()),
    )
    monkeypatch.setattr(
        run_orchestrator,
        "PreferenceMatrix",
        SimpleNamespace(from_yaml=lambda _path: object()),
    )
    monkeypatch.setattr(
        run_orchestrator,
        "_prepare_simulator_workspace",
        lambda **_kwargs: tmp_path / "sim_profile",
    )
    monkeypatch.setattr(
        run_orchestrator,
        "load_simulator_profile",
        lambda root: SimpleNamespace(
            root=root,
            display_name="A",
            load_identity=lambda: object(),
            load_matrix=lambda: object(),
        ),
    )
    monkeypatch.setattr(
        run_orchestrator,
        "UserMemoryStore",
        lambda root, user_name: SimpleNamespace(append_history=lambda *a, **k: None),
    )

    args = SimpleNamespace(
        verbose=True,
        plan=str(plan_path),
        persona_id="user_a",
        memory_condition="no_memory",
        pa_model=model,
        consolidation_model=None,
        memory_side_model=None,
        memos_api_base=None,
        assistant_llm_provider_name="openrouter",
        assistant_llm_api_base=None,
    )
    with pytest.raises(StopBeforeRun):
        await run_orchestrator._execute_benchmark_run(args)

    factory = captured["factory"]
    step = plan.steps[0]

    mainline = factory(step, object(), SimpleNamespace())
    branch_workspace = tmp_path / "branch" / "pa_workspace" / "user_a"
    branch_workspace.mkdir(parents=True)
    branch_recorder = MemoryIORecorder(
        tmp_path / "branch" / "memory_io.jsonl", run_id=plan.run_id
    )
    branch_adapter = object()
    injected = factory(
        step,
        object(),
        SimpleNamespace(),
        workspace=branch_workspace,
        session_manager=run_orchestrator.SessionManager(branch_workspace),
        memory_adapter=branch_adapter,
        memory_backend_namespace=None,
        memory_io_recorder=branch_recorder,
    )

    assert injected.workspace == branch_workspace
    assert injected.memory_adapter is branch_adapter
    assert injected.memory_io_recorder is branch_recorder
    # The mainline values are untouched by the injection.
    assert mainline.workspace != branch_workspace
    assert mainline.memory_io_recorder is not branch_recorder


async def _captured_orchestrator_config(
    tmp_path, monkeypatch, authoring_root, extra_args: dict
):
    """The `OrchestratorConfig` the entry point builds from these arguments.

    Driven through `_execute_benchmark_run` rather than constructed here: the
    question is whether a flag reaches the config at all, and that is exactly
    what a direct construction cannot answer.
    """
    model = "openai/gpt-5.5"
    plan = run_plan_builder._build_frozen_run_plan(
        run_id="borrowroot_run_001",
        persona_id="user_a",
        memory_condition="no_memory",
        pa_model=model,
        script_names=["acc_001"],
    )
    plan_path = plan.dump_yaml(tmp_path / "run_plan.yaml")
    monkeypatch.setattr(
        run_orchestrator,
        "_resolve_assistant_llm_provider_profile",
        lambda _: run_orchestrator.AssistantLLMProviderProfile(
            provider_name="openrouter", api_key="api-key"
        ),
    )
    fixture = authoring_root / "data" / "state_fixtures" / "user_a"
    fixture.mkdir(parents=True, exist_ok=True)
    (fixture / "base.txt").write_text("base", encoding="utf-8")

    captured: dict = {}

    class StopBeforeRun(RuntimeError):
        pass

    def capture(**kwargs):
        captured["config"] = kwargs["config"]
        raise StopBeforeRun

    monkeypatch.setattr(run_orchestrator, "Orchestrator", capture)
    monkeypatch.setattr(
        run_orchestrator, "SessionExecutor", lambda loop_factory: object()
    )
    monkeypatch.setattr(
        run_orchestrator,
        "PersonaIdentity",
        SimpleNamespace(from_yaml=lambda _path: object()),
    )
    monkeypatch.setattr(
        run_orchestrator,
        "PreferenceMatrix",
        SimpleNamespace(from_yaml=lambda _path: object()),
    )
    monkeypatch.setattr(
        run_orchestrator,
        "_prepare_simulator_workspace",
        lambda **_kwargs: tmp_path / "sim_profile",
    )
    monkeypatch.setattr(
        run_orchestrator,
        "load_simulator_profile",
        lambda root: SimpleNamespace(
            root=root,
            display_name="A",
            load_identity=lambda: object(),
            load_matrix=lambda: object(),
        ),
    )
    monkeypatch.setattr(
        run_orchestrator,
        "UserMemoryStore",
        lambda root, user_name: SimpleNamespace(
            append_history=lambda *a, **k: None
        ),
    )

    args = SimpleNamespace(
        verbose=True,
        plan=str(plan_path),
        persona_id="user_a",
        memory_condition="no_memory",
        pa_model=model,
        consolidation_model=None,
        memory_side_model=None,
        memos_api_base=None,
        assistant_llm_provider_name="openrouter",
        assistant_llm_api_base=None,
        **extra_args,
    )
    with pytest.raises(StopBeforeRun):
        await run_orchestrator._execute_benchmark_run(args)
    return captured["config"]


def test_the_borrowed_runs_root_flag_parses_and_defaults_to_nothing() -> None:
    parser = run_orchestrator._build_arg_parser()

    assert parser.parse_args([]).borrowed_runs_root is None
    assert parser.parse_args(
        ["--borrowed-runs-root", "/srv/borrowed/runs"]
    ).borrowed_runs_root == Path("/srv/borrowed/runs")


@pytest.mark.asyncio
async def test_a_given_borrowed_runs_root_reaches_the_orchestrator(
    tmp_path, monkeypatch, authoring_root
):
    """`OrchestratorConfig.borrowed_runs_root` existed with nothing able to set
    it: `harness/orchestrator.py` defines it and reads it, and no third call
    site in the tree passed it, so the borrow silently always used the
    same-parent fallback."""
    borrowed = tmp_path / "MemPA-runs" / "runs"

    config = await _captured_orchestrator_config(
        tmp_path,
        monkeypatch,
        authoring_root,
        {"borrowed_runs_root": borrowed},
    )

    assert config.borrowed_runs_root == borrowed


@pytest.mark.asyncio
async def test_no_borrowed_runs_root_leaves_the_same_parent_fallback(
    tmp_path, monkeypatch, authoring_root
):
    """Adding the flag changes nothing for a launch that omits it: the config
    field stays None, which `Orchestrator._run_one_probe` reads as
    `run_dir.parent`."""
    config = await _captured_orchestrator_config(
        tmp_path, monkeypatch, authoring_root, {}
    )

    assert config.borrowed_runs_root is None


def test_soul_text_forbids_claiming_a_file_change_with_no_write() -> None:
    """acc_006: three merges reported as done on a turn with zero tool calls.

    The claim was then consolidated into HISTORY.md and MEMORY.md as fact.
    """
    soul = run_orchestrator.SOUL_TEXT

    assert soul.startswith("You are a helpful personal assistant.")
    assert (
        'Say a file changed only when a writing tool returned status "updated" '
        "in the same turn; otherwise say what you would change and ask." in soul
    )


# --- the simulator's dictation check is armed from the resolved gate


def _fixture_script():
    return run_plan_builder._load_script(
        Path("tests/fixtures/authoring/normal_session_acc_001.yaml")
    )


def _gate(**overrides) -> dict:
    """A resolved gate for the acc_001 fixture, revealed on its last beat.

    The cue is the one the fixture's outline entry carries
    (`tests/test_normal_session_authoring.py:_gate`): `ex_open_request` has the
    user state the practical decision, which is the reason the PA has anything
    to ask about.
    """
    return {
        "tool": "document_update",
        "param": "replacement",
        "task_fact": "placement_note",
        "dictated_text": "the note the user dictates",
        "key_terms": ["note", "dictates"],
        "gate_cue": "state the practical decision",
        "reveal_beat": "ex_place_the_paper",
        "pre_reveal_write_tools": [],
    } | overrides


def test_a_resolved_gate_arms_the_dictation_check_on_the_script() -> None:
    script = _fixture_script()
    beat = script.beats[-1].beat
    armed = run_orchestrator._arm_dictation_check(script, _gate(reveal_beat=beat))
    assert armed.gate_key_terms == ["note", "dictates"]
    assert armed.gate_dictated_text == "the note the user dictates"
    assert armed.gate_reveal_beat == beat
    # The cue rides along with the beat that owes it, so the actor keeps saying
    # it however the style retry rewrites that turn.
    assert armed.gate_cue == "state the practical decision"
    assert armed.gate_cue_beat == "ex_open_request"
    assert script.gate_key_terms == []
    assert script.gate_cue is None and script.gate_cue_beat is None


def test_the_authorised_tool_is_armed_beside_the_gate_fields() -> None:
    armed = run_orchestrator._arm_dictation_check(
        _fixture_script(), _gate(), "appointment_create"
    )
    assert armed.authorised_tool == "appointment_create"


def test_the_authorised_tool_is_armed_on_a_session_with_no_gate() -> None:
    """A compressed session has no dictation to check and still has a tool the
    user releases, so this one is armed off the draft either way."""
    armed = run_orchestrator._arm_dictation_check(
        _fixture_script(), None, "appointment_create"
    )
    assert armed.authorised_tool == "appointment_create"
    assert armed.gate_key_terms == [] and armed.gate_reveal_beat is None
    assert armed.gate_cue is None and armed.gate_cue_beat is None


def test_a_draft_that_releases_no_step_arms_no_authorised_tool() -> None:
    armed = run_orchestrator._arm_dictation_check(_fixture_script(), _gate())
    assert armed.authorised_tool is None


def test_the_first_beat_before_the_reveal_that_carries_the_cue_owes_it() -> None:
    """Two beats say "keep the work moving"; the earliest is the one armed —
    the cue has to reach the PA before it has to ask, not just before the
    reveal."""
    script = _fixture_script()
    armed = run_orchestrator._arm_dictation_check(
        script, _gate(gate_cue="keep the work moving")
    )
    assert armed.gate_cue_beat == "ex_frame_scope"


def test_a_cue_only_in_the_framing_prose_arms_no_beat() -> None:
    """acc_023's shape: the scenario is handed to the PA, never said by the
    actor, so there is no beat to hold to and the cue still arms."""
    armed = run_orchestrator._arm_dictation_check(
        _fixture_script(), _gate(gate_cue="in his WIT office in early October")
    )
    assert armed.gate_cue == "in his WIT office in early October"
    assert armed.gate_cue_beat is None


def test_the_draft_s_own_cue_beat_wins_over_the_search() -> None:
    """Design B: the author pins the beat, and the validator held it to sitting
    before the reveal. The search is what could not find it — the cue lives in
    the scenario prose while the beats paraphrase it."""
    armed = run_orchestrator._arm_dictation_check(
        _fixture_script(),
        _gate(gate_cue="in his WIT office in early October"),
        None,
        "ex_frame_scope",
    )
    assert armed.gate_cue_beat == "ex_frame_scope"


def test_a_declared_beat_the_script_does_not_have_falls_back_to_the_search() -> None:
    armed = run_orchestrator._arm_dictation_check(
        _fixture_script(), _gate(gate_cue="keep the work moving"), None, "b9_nowhere"
    )
    assert armed.gate_cue_beat == "ex_frame_scope"


def test_the_declared_cue_beat_is_read_off_the_draft_file(tmp_path) -> None:
    draft = tmp_path / "acc_003.yaml"
    draft.write_text("gate_cue_beat: b3_frame_verification_scope\nbeats: []\n", encoding="utf-8")
    assert run_orchestrator._declared_cue_beat(draft) == "b3_frame_verification_scope"

    forced = tmp_path / "evolv_01.yaml"
    forced.write_text("beats: []\n", encoding="utf-8")
    assert run_orchestrator._declared_cue_beat(forced) is None


def test_a_draft_declaring_no_cue_beat_still_resolves_by_search() -> None:
    """The forced line (`evolv_01`) carries no such field and resolves as before."""
    armed = run_orchestrator._arm_dictation_check(
        _fixture_script(), _gate(gate_cue="keep the work moving"), None, None
    )
    assert armed.gate_cue_beat == "ex_frame_scope"


def test_an_opening_reveal_beat_owes_the_cue_it_carries_before_the_hand_over() -> None:
    """acc_009's shape: the opening beat IS the probe beat, so its own
    instruction counts — but only up to the sentence handing the fact over,
    which is why "opens the wrong material" (said after the `task_facts`
    reference) arms no beat."""
    script = _fixture_script()
    opening = script.beats[0].beat
    armed = run_orchestrator._arm_dictation_check(script, _gate(reveal_beat=opening))
    assert armed.gate_cue_beat == opening
    after_the_hand_over = run_orchestrator._arm_dictation_check(
        script, _gate(reveal_beat=opening, gate_cue="opens the wrong material")
    )
    assert after_the_hand_over.gate_cue_beat is None


def test_no_gate_leaves_the_dictation_check_unarmed() -> None:
    """`authorised_tool` is armed for every session, so a gateless script is now
    re-validated rather than handed straight back; nothing about the dictation
    check is armed by that pass."""
    script = _fixture_script()
    armed = run_orchestrator._arm_dictation_check(script, None)
    assert armed.gate_key_terms == [] and armed.gate_reveal_beat is None
    assert armed.gate_cue is None and armed.gate_cue_beat is None
    assert script.gate_key_terms == [] and script.gate_reveal_beat is None
    assert script.gate_cue is None and script.gate_cue_beat is None


def test_a_gate_naming_a_beat_the_script_lacks_is_refused() -> None:
    # The cue finder hands the missing beat on rather than raising over it, so
    # the error still names the field that is actually wrong.
    with pytest.raises(Exception, match="gate_reveal_beat"):
        run_orchestrator._arm_dictation_check(
            _fixture_script(), _gate(reveal_beat="no_such_beat")
        )


@pytest.mark.asyncio
async def test_the_loop_carries_the_cue_off_the_beat_the_simulator_skips(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Live `acc_003`: the beat the cue was armed on was skipped after
    a turn limit and the cue never reached the PA. The loop installs the carry
    hook on the writer, so the skip moves the obligation onto the landing beat
    of the very script the simulator is holding."""
    captured: dict = {}

    class FakeBackend:
        async def drain_memory_tasks(self) -> None:
            return None

    class FakeSimulatorConfig:
        model = "sim-model"

        def create_provider(self):
            return object()

        def create_backend(self, provider, *, profile, output_contract_retries):
            return FakeBackend()

    class FakePA:
        async def close_mcp(self) -> None:
            return None

    class CapturingSimulatorLoop:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        async def run_session(self):
            return []

    class RecordingWriter:
        def __init__(self):
            self.hooks = []

        def add_event_hook(self, hook):
            self.hooks.append(hook)

    script = _fixture_script()
    monkeypatch.setattr(run_plan_builder, "_load_script", lambda _: script)
    monkeypatch.setattr(
        run_orchestrator, "build_pa_from_provider_profile", lambda **_: FakePA()
    )
    monkeypatch.setattr(run_orchestrator, "RecordingPA", lambda pa: pa)
    monkeypatch.setattr(run_orchestrator, "SimulatorLoop", CapturingSimulatorLoop)
    monkeypatch.setattr(
        run_orchestrator, "SessionEndSelfReporter", lambda *a, **k: object()
    )

    writer = RecordingWriter()
    loop = run_orchestrator.RealSessionLoop(
        step=SimpleNamespace(
            # `_load_script` is faked below, but `authorised_tool` is resolved
            # off the draft itself, so the path has to name a real one.
            script_path=Path("tests/fixtures/authoring/normal_session_acc_001.yaml"),
            # An accumulation session, so its PA requires committed writes.
            kind="accumulation",
            persona_id="user_a",
            benchmark_session_timestamp=datetime(
                2026, 10, 1, 9, 0, tzinfo=timezone.utc
            ),
        ),
        transcript_writer=writer,
        stage_runtime=SimpleNamespace(mcp_server_config=lambda: {}, gate=_gate(),
                                      state_root=tmp_path, session_date=None, session_timestamp=None),
        identity=object(),
        matrix=object(),
        sim_config=FakeSimulatorConfig(),
        sim_profile=SimpleNamespace(
            root=tmp_path / "sim-profile", build_context=lambda: "profile context"
        ),
        workspace=tmp_path,
        pa_session_key="run_001_main",
        benchmark_run_id="run_001",
        assistant_llm_provider_profile=SimpleNamespace(),
        pa_model="pa-model",
        pa_max_tokens=8192,
        memory_adapter=object(),
        consolidation_llm_provider=None,
        consolidation_model=None,
        memory_io_recorder=object(),
        memory_retrieval_max_tokens=1234,
        memory_retrieval_max_items=6,
        simulator_runtime=SimulatorRuntimeConfig(
            output_contract_retries=2,
            self_report_contract_retries=7,
            simulator_model="sim-model",
        ),
    )

    assert await loop.run_session() == []
    armed = captured["script"]
    assert armed.gate_cue_beat == "ex_open_request"
    assert len(writer.hooks) == 1

    writer.hooks[0](
        {
            "kind": "beat_skipped",
            "beat": "ex_open_request",
            "reason": "jump_to_preference_beat_after_turn_limit",
        }
    )

    assert armed.gate_cue_beat == armed.beats[1].beat


def _forced_plan_kwargs() -> dict:
    return dict(
        run_id="fe_digest_run_001",
        persona_id="user_a",
        memory_condition="no_memory",
        pa_model="openai/gpt-5.5",
        script_names=["acc_075"],
    )


def test_a_forced_transcript_its_validator_refuses_is_not_frozen(
    authoring_root, monkeypatch
):
    """The forced-event transcript is checked against its sources at freeze.

    The real check, not the stand-in: the synthetic draft the transcript names
    holds only `event_meta.event_id` and `source_acc_num`, so it is not a valid
    forced-event draft and the freeze refuses the transcript.
    """
    monkeypatch.setattr(
        run_plan_builder, "_validate_forced_transcript", REAL_VALIDATE_FORCED_TRANSCRIPT
    )
    with pytest.raises(
        ValueError,
        match="(?s)does not pass its validator.*source forced event draft is invalid",
    ):
        run_plan_builder._build_frozen_run_plan(**_forced_plan_kwargs())


async def _launch_until_the_ledger(tmp_path, monkeypatch, *, acc_range):
    """Launch the synthetic probe persona through `_execute_benchmark_run` and
    stop at the ledger, the first thing built after the plan, its GT copy and
    the `freeze` record land. Returns the builder calls and the run dir."""
    from test_harness.test_run_plan_builder_probes import (
        PERSONA,
        _write_acc_line,
        _write_probe_line,
    )

    _write_acc_line(tmp_path)
    _write_probe_line(tmp_path)
    monkeypatch.setattr(run_plan_builder, "PROJECT_ROOT", tmp_path)
    run_dir = tmp_path / "workspace/runs/probe_run"
    monkeypatch.setattr(run_orchestrator, "_run_dir", lambda **_kwargs: run_dir)
    monkeypatch.setattr(
        run_orchestrator,
        "_resolve_assistant_llm_provider_profile",
        lambda _: run_orchestrator.AssistantLLMProviderProfile(
            provider_name="openrouter", api_key="api-key"
        ),
    )
    build_calls: list[dict] = []
    real_build = run_plan_builder._build_frozen_run_plan

    def recording_build(**kwargs):
        build_calls.append(kwargs)
        return real_build(**kwargs)

    monkeypatch.setattr(run_plan_builder, "_build_frozen_run_plan", recording_build)

    class StopAfterFreeze(RuntimeError):
        pass

    def stop_after_freeze(*_args, **_kwargs):
        raise StopAfterFreeze

    # The ledger is the first thing built after the plan and its copy land.
    monkeypatch.setattr(run_orchestrator, "RunLedger", stop_after_freeze)
    args = SimpleNamespace(
        verbose=True,
        plan=None,
        run_id="probe_run",
        persona_id=PERSONA,
        acc_range=acc_range,
        acc_list=None,
        memory_condition="no_memory",
        pa_model="openai/gpt-5.5",
        consolidation_model=None,
        memory_side_model=None,
        memos_api_base=None,
        regenerate_plan=False,
        overwrite=False,
        assistant_llm_provider_name="openrouter",
        assistant_llm_api_base=None,
    )

    with pytest.raises(StopAfterFreeze):
        await run_orchestrator._execute_benchmark_run(args)
    return build_calls, run_dir


def _freeze_record(run_dir):
    records = yaml.safe_load((run_dir / "run_info.yaml").read_text(encoding="utf-8"))
    assert [record["event"] for record in records] == ["freeze"]
    return records[0]


@pytest.mark.asyncio
async def test_a_fresh_plan_freezes_the_persona_probe_line_and_its_gt_copy(
    tmp_path, monkeypatch
):
    """A persona with a probe outline and resolved GT gets
    its probes frozen into the plan, and the GT copy lands beside it."""
    from test_harness.test_run_plan_builder_probes import ACC_COUNT, PERSONA

    build_calls, run_dir = await _launch_until_the_ledger(
        tmp_path, monkeypatch, acc_range=f"1-{ACC_COUNT}"
    )

    assert [call["with_probes"] for call in build_calls] == [True]
    plan = load_run_plan(run_dir / "run_plan.yaml", project_root=tmp_path)
    assert len(plan.steps) == ACC_COUNT + 33
    source = tmp_path / f"Authoring/probe/outline/{PERSONA}/resolved_gt.yaml"
    assert (run_dir / "resolved_gt.yaml").read_bytes() == source.read_bytes()
    assert plan.resolved_gt is not None
    freeze = _freeze_record(run_dir)
    assert freeze["acc_range"] == f"1-{ACC_COUNT}"
    assert freeze["probe_step_ids"] == [
        step.step_id for step in plan.steps if step.kind in PROBE_KINDS
    ]
    assert len(freeze["probe_step_ids"]) == 33


@pytest.mark.asyncio
async def test_a_short_range_freezes_only_its_probes_and_records_them(
    tmp_path, monkeypatch
):
    """`--acc-range 1-30` on a persona with a probe line freezes, with only the
    two pre probes cut inside it (before acc_010 and acc_030), and the freeze
    record names the range and those two probes."""
    _build_calls, run_dir = await _launch_until_the_ledger(
        tmp_path, monkeypatch, acc_range="1-30"
    )

    plan = load_run_plan(run_dir / "run_plan.yaml", project_root=tmp_path)
    assert [step.step_id for step in plan.steps if step.kind in PROBE_KINDS] == [
        "pre_01", "pre_02",
    ]
    freeze = _freeze_record(run_dir)
    assert freeze["acc_range"] == "1-30"
    assert freeze["probe_step_ids"] == ["pre_01", "pre_02"]


# --- attaching a probe plan to a finished accumulation run -------------

_ATTACH_RUN_ID = "attach_probes"


class _StopAtClient(RuntimeError):
    pass


def _stage_attach_probes(tmp_path, monkeypatch):
    """The synthetic persona and run directory for the "attach probes" path."""
    from test_harness.test_run_plan_builder_probes import (
        ACC_COUNT,
        PERSONA,
        _write_acc_line,
    )

    _write_acc_line(tmp_path)
    monkeypatch.setattr(run_plan_builder, "PROJECT_ROOT", tmp_path)
    run_dir = tmp_path / f"workspace/runs/{_ATTACH_RUN_ID}"
    monkeypatch.setattr(run_orchestrator, "_run_dir", lambda **_kwargs: run_dir)
    monkeypatch.setattr(
        run_orchestrator,
        "_resolve_assistant_llm_provider_profile",
        lambda _: run_orchestrator.AssistantLLMProviderProfile(
            provider_name="openrouter", api_key="api-key"
        ),
    )
    monkeypatch.setenv("OPENAI_EMB_API_KEY", "embedding-key")
    return PERSONA, ACC_COUNT, run_dir


def _attach_args(persona_id, acc_count, *, regenerate_plan):
    return SimpleNamespace(
        verbose=True,
        plan=None,
        run_id=_ATTACH_RUN_ID,
        persona_id=persona_id,
        acc_range=f"1-{acc_count}",
        acc_list=None,
        memory_condition="simple_rag",
        pa_model="openai/gpt-5.5",
        consolidation_model=None,
        memory_side_model=None,
        memos_api_base=None,
        regenerate_plan=regenerate_plan,
        overwrite=False,
        assistant_llm_provider_name="openrouter",
        assistant_llm_api_base=None,
    )


def _record_client_kwargs(monkeypatch) -> list[dict]:
    """Stop the run at the mem0-shaped client, holding on to its arguments."""
    calls: list[dict] = []

    def factory(**kwargs):
        calls.append(kwargs)
        raise _StopAtClient

    monkeypatch.setattr(run_orchestrator, "create_simple_rag_mem0_client", factory)
    return calls


def _mark_timeline_done(run_dir, acc_count) -> None:
    ledger = RunLedger(run_dir, _ATTACH_RUN_ID)
    for acc_num in range(1, acc_count + 1):
        ledger.mark_done(f"acc_{acc_num:03d}")


@pytest.mark.asyncio
async def test_regenerating_a_plan_over_a_recorded_ledger_resumes_that_run(
    tmp_path, monkeypatch
):
    """A run executes its timeline now and gets its probes attached later.

    `--regenerate-plan` on a directory whose ledger already records steps is a
    resume of that run, so the Qdrant collection and the history db those steps
    wrote are expected to be there -- and the plan is still rebuilt and
    re-frozen, probes and ground-truth copy included.
    """
    from test_harness.test_run_plan_builder_probes import _write_probe_line

    persona_id, acc_count, run_dir = _stage_attach_probes(tmp_path, monkeypatch)
    _write_probe_line(tmp_path)
    _mark_timeline_done(run_dir, acc_count)
    calls = _record_client_kwargs(monkeypatch)

    with pytest.raises(_StopAtClient):
        await run_orchestrator._execute_benchmark_run(
            _attach_args(persona_id, acc_count, regenerate_plan=True)
        )

    assert calls[0]["allow_existing_collection"] is True
    assert calls[0]["allow_existing_history_db"] is True
    assert calls[0]["require_existing_history_db"] is True
    assert calls[0]["drop_existing_collection"] is False

    plan = load_run_plan(run_dir / "run_plan.yaml", project_root=tmp_path)
    assert len(plan.steps) == acc_count + 33
    assert [step.step_id for step in plan.steps if step.kind in PROBE_KINDS] == (
        [f"pre_0{index}" for index in range(1, 6)]
        + [f"fin_{index:02d}" for index in range(1, 29)]
    )
    assert (run_dir / "resolved_gt.yaml").is_file()


@pytest.mark.asyncio
async def test_regenerating_a_plan_on_an_empty_run_directory_is_not_a_resume(
    tmp_path, monkeypatch
):
    """No recorded step means no backend state to reuse: the refusals stand."""
    from test_harness.test_run_plan_builder_probes import _write_probe_line

    persona_id, acc_count, run_dir = _stage_attach_probes(tmp_path, monkeypatch)
    _write_probe_line(tmp_path)
    assert not run_dir.exists()
    calls = _record_client_kwargs(monkeypatch)

    with pytest.raises(_StopAtClient):
        await run_orchestrator._execute_benchmark_run(
            _attach_args(persona_id, acc_count, regenerate_plan=True)
        )

    assert calls[0]["allow_existing_collection"] is False
    assert calls[0]["allow_existing_history_db"] is False
    assert calls[0]["require_existing_history_db"] is False


@pytest.mark.asyncio
async def test_a_regenerated_plan_keeps_the_frozen_timeline_step_ids(
    tmp_path, monkeypatch
):
    """The ledger's `done` marks only apply if the timeline is the same one.

    The probe steps merge in around it, so the timeline step ids and their
    order have to survive the rebuild untouched.
    """
    from test_harness.test_run_plan_builder_probes import _write_probe_line

    persona_id, acc_count, run_dir = _stage_attach_probes(tmp_path, monkeypatch)

    # The finished run, frozen before the persona had a probe line at all.
    class _StopAtLedger(RuntimeError):
        pass

    with monkeypatch.context() as first_run:
        def stop_at_ledger(*_args, **_kwargs):
            raise _StopAtLedger

        first_run.setattr(run_orchestrator, "RunLedger", stop_at_ledger)
        with pytest.raises(_StopAtLedger):
            await run_orchestrator._execute_benchmark_run(
                _attach_args(persona_id, acc_count, regenerate_plan=False)
            )
    frozen = load_run_plan(run_dir / "run_plan.yaml", project_root=tmp_path)
    assert [step.kind for step in frozen.steps] == ["accumulation"] * acc_count

    _write_probe_line(tmp_path)
    _mark_timeline_done(run_dir, acc_count)
    _record_client_kwargs(monkeypatch)

    with pytest.raises(_StopAtClient):
        await run_orchestrator._execute_benchmark_run(
            _attach_args(persona_id, acc_count, regenerate_plan=True)
        )

    rewritten = load_run_plan(run_dir / "run_plan.yaml", project_root=tmp_path)
    assert [
        step.step_id for step in rewritten.steps if step.kind not in PROBE_KINDS
    ] == [step.step_id for step in frozen.steps]


def _memos_attach_args(persona_id, acc_count, *, regenerate_plan):
    args = _attach_args(persona_id, acc_count, regenerate_plan=regenerate_plan)
    args.memory_condition = "memos"
    args.memory_side_model = "openai/gpt-5.5"
    args.memos_api_base = "http://127.0.0.1:8100"
    return args


@pytest.mark.asyncio
async def test_re_freezing_a_finished_memos_plan_keeps_its_cube_id(
    tmp_path, monkeypatch
):
    """The attach writes into the cube the timeline already filled.

    `memos_cube_id` is minted per plan, so re-freezing used to point the
    mainline at a brand-new, empty cube while the checkpoints still named the
    old one -- and `_validate_memos_backend_state` then reported every audited
    memory missing. It is not an input to the run-config hash, so carrying it
    over leaves the run's identity where it was.
    """
    from test_harness.test_run_plan_builder_probes import _write_probe_line

    persona_id, acc_count, run_dir = _stage_attach_probes(tmp_path, monkeypatch)

    class _StopAtLedger(RuntimeError):
        pass

    with monkeypatch.context() as first_run:

        def stop_at_ledger(*_args, **_kwargs):
            raise _StopAtLedger

        first_run.setattr(run_orchestrator, "RunLedger", stop_at_ledger)
        with pytest.raises(_StopAtLedger):
            await run_orchestrator._execute_benchmark_run(
                _memos_attach_args(persona_id, acc_count, regenerate_plan=False)
            )
    frozen = load_run_plan(run_dir / "run_plan.yaml", project_root=tmp_path)
    assert frozen.memos_cube_id

    _write_probe_line(tmp_path)
    _mark_timeline_done(run_dir, acc_count)

    def stop_at_client(**_kwargs):
        raise _StopAtClient

    monkeypatch.setattr(run_orchestrator, "MemOSHttpClient", stop_at_client)
    with pytest.raises(_StopAtClient):
        await run_orchestrator._execute_benchmark_run(
            _memos_attach_args(persona_id, acc_count, regenerate_plan=True)
        )

    rewritten = load_run_plan(run_dir / "run_plan.yaml", project_root=tmp_path)
    assert rewritten.memos_cube_id == frozen.memos_cube_id
    assert len(rewritten.steps) == acc_count + 33


def _re_freeze_kwargs() -> dict:
    return dict(
        run_id="re_freeze_run_001",
        persona_id="user_a",
        memory_condition="no_memory",
        pa_model="openai/gpt-5.5",
        script_names=["acc_001", "acc_075"],
    )


def test_re_freezing_a_plan_keeps_a_done_forced_step_verbatim(authoring_root):
    """A done step is carried as it ran, not re-frozen from today's artifacts.

    The carried step is not rebuilt at all, so the artifacts behind it are
    never re-read -- here the forced draft is edited after the run, which a
    re-freeze refuses and a carry-over does not notice.
    """
    frozen = run_plan_builder._build_frozen_run_plan(**_re_freeze_kwargs())
    hand_set = frozen.steps[1].model_copy(update={"story_id": "as_it_ran"})
    existing = frozen.model_copy(
        update={"steps": [frozen.steps[0], hand_set]}
    )
    (
        authoring_root / "Authoring/sessions/forced/drafts/user_a/evolv_05.yaml"
    ).write_text(
        "event_meta:\n  event_id: evolv_05\n  note: edited after the run\n",
        encoding="utf-8",
    )

    rebuilt = run_plan_builder._build_frozen_run_plan(
        **_re_freeze_kwargs(),
        existing_plan=existing,
        done_step_ids=frozenset({"acc_001", "evolv_05"}),
    )

    assert [step.step_id for step in rebuilt.steps] == ["acc_001", "evolv_05"]
    assert rebuilt.steps[0] == frozen.steps[0]
    assert rebuilt.steps[1] == hand_set


def test_re_freezing_a_plan_refuses_a_done_step_that_changed(authoring_root):
    """Carrying a step over is only sound while it is the same step."""
    frozen = run_plan_builder._build_frozen_run_plan(**_re_freeze_kwargs())
    moved = frozen.steps[0].model_copy(
        update={
            "script_path": Path(
                "Authoring/sessions/normal/drafts/user_a/acc_002.yaml"
            )
        }
    )
    existing = frozen.model_copy(update={"steps": [moved, frozen.steps[1]]})

    with pytest.raises(ValueError, match="already finished"):
        run_plan_builder._build_frozen_run_plan(
            **_re_freeze_kwargs(),
            existing_plan=existing,
            done_step_ids=frozenset({"acc_001"}),
        )


def test_re_freezing_a_plan_rebuilds_the_probe_steps_it_is_for(
    tmp_path, monkeypatch
):
    """Probes are the reason to regenerate, so nothing about them is carried:
    they are rebuilt from the drafts as they are now, whatever the previous
    plan froze and whatever the ledger says about it.
    """
    from test_harness.test_run_plan_builder_probes import (
        ACC_COUNT,
        PERSONA,
        _write_acc_line,
        _write_probe_line,
    )

    _write_acc_line(tmp_path)
    _write_probe_line(tmp_path)
    monkeypatch.setattr(run_plan_builder, "PROJECT_ROOT", tmp_path)
    kwargs = dict(
        run_id="re_freeze_probe_run",
        persona_id=PERSONA,
        memory_condition="no_memory",
        pa_model="openai/gpt-5.5",
        script_names=[f"acc_{num:03d}" for num in range(1, ACC_COUNT + 1)],
        with_probes=True,
    )
    frozen = run_plan_builder._build_frozen_run_plan(**kwargs)
    existing = frozen.model_copy(
        update={
            "steps": [
                step.model_copy(update={"story_id": "stale"})
                if step.kind in PROBE_KINDS
                else step
                for step in frozen.steps
            ]
        }
    )

    rebuilt = run_plan_builder._build_frozen_run_plan(
        **kwargs,
        existing_plan=existing,
        # Even marked done, a probe step is rebuilt.
        done_step_ids=frozenset(step.step_id for step in frozen.steps),
    )

    assert [step.step_id for step in rebuilt.steps] == [
        step.step_id for step in frozen.steps
    ]
    assert rebuilt.steps == frozen.steps
    assert "stale" not in {step.story_id for step in rebuilt.steps}


# --- `--acc-list`: naming a persona's whole, non-contiguous timeline ---------

#: `user_b`'s 99 Outline sessions, as contiguous runs. Derived from the
#: Outline, never typed by hand; `test_acc_list_names_user_b_whole_timeline`
#: recomputes it and fails if the Outline moves.
USER_B_ACC_LIST = (
    "1-15,17-31,33-34,36-49,51-59,61-65,67-77,79-99,101-103,105-108"
)


def _acc_list_from_outline(persona_id: str) -> str:
    """Contiguous-run spelling of every acc the persona's Outline declares."""
    outline_path = (
        run_plan_builder.PROJECT_ROOT
        / f"Authoring/outline/drafts/{persona_id}/outline.yaml"
    )
    outline = yaml.safe_load(outline_path.read_text(encoding="utf-8"))
    nums = sorted(
        session["identity"]["acc_num"] for session in outline["sessions"]
    )
    runs: list[list[int]] = []
    for num in nums:
        if runs and num == runs[-1][-1] + 1:
            runs[-1].append(num)
        else:
            runs.append([num])
    return ",".join(
        str(run[0]) if len(run) == 1 else f"{run[0]}-{run[-1]}" for run in runs
    )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("7", ["acc_007"]),
        ("3-5", ["acc_003", "acc_004", "acc_005"]),
        ("1,4-6,9", ["acc_001", "acc_004", "acc_005", "acc_006", "acc_009"]),
        (" 2 , 4 ", ["acc_002", "acc_004"]),
        ("9-9", ["acc_009"]),
        # de-duplicated, and kept in the order the flag names them
        ("3,1-3,2", ["acc_003", "acc_001", "acc_002"]),
    ],
)
def test_acc_list_expands_integers_and_inclusive_ranges(value, expected):
    assert run_orchestrator._acc_names_from_list(value) == expected


@pytest.mark.parametrize(
    "value",
    ["", "   ", "bad", "1-", "-3", "3-2", "0", "0-2", "1,,2", "1..3", "acc_001"],
)
def test_acc_list_refuses_a_bad_token_as_an_argparse_error(value):
    parser = run_orchestrator._build_arg_parser()
    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(["--acc-list", value])
    assert excinfo.value.code == 2


def test_acc_list_and_acc_range_are_mutually_exclusive():
    parser = run_orchestrator._build_arg_parser()
    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(["--acc-range", "1-3", "--acc-list", "5"])
    assert excinfo.value.code == 2


def test_neither_acc_flag_keeps_the_acc_range_default():
    """Adding the flag changes no default: `--acc-range` still stands in."""
    args = run_orchestrator._build_arg_parser().parse_args([])
    assert args.acc_list is None
    assert args.acc_range is None
    selected = (
        args.acc_list
        if args.acc_list
        else run_orchestrator._acc_names(
            args.acc_range or run_orchestrator.DEFAULT_ACC_RANGE
        )
    )
    assert selected == [f"acc_{i:03d}" for i in range(1, 11)]


def test_acc_list_names_user_b_whole_timeline():
    """The 99-session timeline no contiguous `--acc-range` can express.

    `user_b`'s Outline runs `acc_001`..`acc_108` with nine probe-slot gaps, so
    this is the flag's reason to exist. The plan is built offline and costs
    nothing: it reads the Outline and the drafts, and makes no API call.
    """
    assert _acc_list_from_outline("user_b") == USER_B_ACC_LIST

    parsed = run_orchestrator._build_arg_parser().parse_args(
        ["--acc-list", USER_B_ACC_LIST]
    )
    assert len(parsed.acc_list) == 99

    plan = run_plan_builder._build_frozen_run_plan(
        run_id="acc_list_user_b_timeline",
        persona_id="user_b",
        memory_condition="no_memory",
        pa_model="deepseek/deepseek-v4-flash",
        script_names=parsed.acc_list,
    )
    kinds = [step.kind for step in plan.steps]
    assert len(plan.steps) == 99
    assert kinds.count("accumulation") == 94
    assert kinds.count("forced_event") == 5

    # The same 99 names spelled as ten separate contiguous ranges freeze the
    # same steps.
    from_ranges: list[str] = []
    for segment in USER_B_ACC_LIST.split(","):
        from_ranges.extend(
            run_orchestrator._acc_names(segment)
            if "-" in segment
            else [f"acc_{int(segment):03d}"]
        )
    assert from_ranges == parsed.acc_list
    range_plan = run_plan_builder._build_frozen_run_plan(
        run_id="acc_list_user_b_timeline",
        persona_id="user_b",
        memory_condition="no_memory",
        pa_model="deepseek/deepseek-v4-flash",
        script_names=from_ranges,
    )
    assert [step.step_id for step in range_plan.steps] == [
        step.step_id for step in plan.steps
    ]


def test_acc_list_still_fails_on_an_acc_the_outline_lacks():
    """No skip-missing behaviour: the builder's refusal is unchanged.

    The acc has to sit outside the `1..FULL_TIMELINE_ACC_COUNT` skeleton to be
    a hole at all. Inside it, a complete Outline's silence is one of the nine
    gap days and the builder emits a placeholder step for it instead
    (`harness/run_plan_builder.py`, `_placeholder_step`) -- that is the gap-day
    contract, not something `--acc-list` introduces.
    """
    names = run_orchestrator._acc_names_from_list("1,109")
    assert names == ["acc_001", "acc_109"]
    with pytest.raises(ValueError, match="Session is missing from the Authoring Outline"):
        run_plan_builder._build_frozen_run_plan(
            run_id="acc_list_missing",
            persona_id="user_b",
            memory_condition="no_memory",
            pa_model="deepseek/deepseek-v4-flash",
            script_names=names,
        )


@pytest.mark.asyncio
async def test_a_pending_recovery_builds_its_client_before_any_recorder(
    tmp_path,
    monkeypatch,
    authoring_root,
):
    """The client that performs the rollback is built; nothing else is yet.

    Two things the plan asks of the startup path. The client a recovery needs
    must not require the live state the recovery exists to rebuild -- a
    recovery interrupted inside the backend layer can leave the history
    database not yet copied back from the snapshot. And the run's recorder and
    adapter belong *after* the recovery, so they are never built on top of
    half-restored data. The ledger here is deliberately clean: nothing about
    it says this run needs anything, and the marker is what does.
    """
    monkeypatch.setenv("OPENAI_EMB_API_KEY", "embedding-key")
    model = "openai/gpt-5.5"
    config = run_orchestrator._effective_mem0_runtime_config("simple_rag")
    assert config is not None
    plan = run_plan_builder._build_frozen_run_plan(
        run_id="pending_run_001",
        persona_id="user_a",
        memory_condition="simple_rag",
        pa_model=model,
        script_names=["acc_001"],
        qdrant_host="frozen-qdrant.internal",
        qdrant_port=6333,
        mem0_runtime_config=config,
    )
    plan_path = plan.dump_yaml(tmp_path / "run_plan.yaml")

    ledger = RunLedger(tmp_path, plan.run_id)
    ledger.mark_running("acc_001")
    ledger.mark_done("acc_001")

    # A rollback that began and did not finish, exactly as `restore` records it.
    marker = tmp_path / "recovery" / "pending.json"
    marker.parent.mkdir(parents=True)
    marker.write_text(
        json.dumps(
            {
                "recovery_id": "20260918T000000000000",
                "target_checkpoint_id": "acc_001",
                "history": ["acc_001"],
                "revoked": [],
                "evidence_dir": "crash_scene/20260918T000000000000",
            }
        ),
        encoding="utf-8",
    )

    client_kwargs: list[dict] = []

    def fake_client_factory(**kwargs):
        client_kwargs.append(kwargs)
        return object()

    monkeypatch.setattr(
        run_orchestrator, "create_simple_rag_mem0_client", fake_client_factory
    )
    monkeypatch.setattr(
        run_orchestrator,
        "_resolve_assistant_llm_provider_profile",
        lambda _: run_orchestrator.AssistantLLMProviderProfile(
            provider_name="openrouter", api_key="test-key"
        ),
    )

    recorders: list = []
    real_recorder = run_orchestrator.MemoryIORecorder
    monkeypatch.setattr(
        run_orchestrator,
        "MemoryIORecorder",
        lambda *args, **kwargs: recorders.append(1) or real_recorder(*args, **kwargs),
    )

    reached: list = []

    def fake_restore(self, checkpoint_id, *, ledger):
        reached.append(
            {
                "checkpoint_id": checkpoint_id,
                "client": self._require_backend_client(),
                "recorders_built": len(recorders),
            }
        )
        raise _RollbackReached(checkpoint_id)

    monkeypatch.setattr(
        run_orchestrator.CheckpointManager, "restore", fake_restore
    )

    args = SimpleNamespace(
        verbose=True,
        plan=str(plan_path),
        persona_id="user_a",
        memory_condition="simple_rag",
        pa_model=model,
        consolidation_model=None,
        memory_side_model=None,
        memos_api_base=None,
        assistant_llm_provider_name="openrouter",
        assistant_llm_api_base=None,
    )

    with pytest.raises(_RollbackReached, match="acc_001"):
        await run_orchestrator._execute_benchmark_run(args)

    # The marker, not the ledger, sent the run back to acc_001.
    assert len(reached) == 1
    assert reached[0]["checkpoint_id"] == "acc_001"
    # Built, and bound to the manager that performs the rollback.
    assert len(client_kwargs) == 1
    assert reached[0]["client"] is not None
    # The live history database is not demanded: it is what the recovery is
    # about to put back. Without a pending marker this ledger would demand it.
    assert client_kwargs[0]["require_existing_history_db"] is False
    # And nothing had been built on the half-restored run yet.
    assert reached[0]["recorders_built"] == 0
