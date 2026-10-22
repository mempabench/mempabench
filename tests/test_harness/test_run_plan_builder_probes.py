"""Probe steps in a frozen RunPlan: constructed from the probe outline and
resolved_gt.yaml, merged by cut point."""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import yaml

from harness import run_plan_builder
from harness.benchmark_clock import SESSION_ENVELOPE_MINUTES
from harness.run_plan import (
    FULL_TIMELINE_ACC_COUNT,
    RunPlanLoadError,
    load_run_plan,
    resolve_probe_overlay,
    validate_full_timeline_shape,
)
from data.personas.coverage import TAXONOMY
from scripts.resolved_gt import setting_for
from scripts.session_authoring_schema import FORCED_EVENT_ID_BY_ACC
from test_harness.test_run_orchestrator import (
    FORCED_EVENT_TRANSCRIPT_YAML,
    NORMAL_SESSION_DRAFT_YAML,
)
PERSONA = "user_x"
ACC_COUNT = 99
# user_b's own gap days and forced-event slots, on the synthetic line.
GAP_ACCS = (16, 32, 35, 50, 60, 66, 78, 100, 104)
FORCED_ACCS = (15, 30, 45, 75, 90)
# The five pre-event cut points and the final they pair with.
PRE_BEFORE_ACC = {"pre_01": 10, "pre_02": 30, "pre_03": 50, "pre_04": 70, "pre_05": 90}
PRE_PAIRS_WITH = {"pre_01": "fin_01", "pre_02": "fin_03", "pre_03": "fin_05",
                  "pre_04": "fin_15", "pre_05": "fin_17"}
ENVELOPE = timedelta(minutes=SESSION_ENVELOPE_MINUTES)


def _acc_clock(acc_num: int) -> datetime:
    """acc_001..099 packed into early_oct: 20 sessions a day, one an hour."""
    return datetime(2026, 10, 1, tzinfo=timezone.utc) + timedelta(
        days=(acc_num - 1) // 20, hours=(acc_num - 1) % 20
    )


def _final_clock() -> datetime:
    return _acc_clock(ACC_COUNT) + ENVELOPE


def _clock(stamp: datetime) -> dict:
    return {"date": stamp.date().isoformat(), "hour": stamp.hour}


def _cell(fin_index: int) -> str:
    """`fin_01..14` -> work × TAXONOMY order, `fin_15..28` -> personal × the same, the
    cell space `scripts/probe_outline_schema.py::final_probe_id` maps back from: the
    plan freeze now reads the pairing out of `resolved_gt.yaml` itself, so the
    synthetic line's cells are the real ones rather than `attr01`-shaped stand-ins."""
    context = "work" if fin_index <= 14 else "personal"
    return f"{context}_{list(TAXONOMY)[(fin_index - 1) % 14]}"


def _write_acc_line(
    root: Path,
    *,
    acc_nums: "tuple[int, ...] | None" = None,
    forced_accs: tuple[int, ...] = (),
) -> None:
    acc_nums = tuple(acc_nums or range(1, ACC_COUNT + 1))
    sessions = [
        {
            "identity": {
                "acc_num": acc_num,
                "session_type": (
                    "evolving_event" if acc_num in forced_accs else "stable"
                ),
                "timeline_slot": "cycle1_work",
                "target_cell": "work_verbosity",
            },
            "story": {
                "domain": "literature_and_citations",
                "story_id": f"story_{acc_num:03d}",
                "time_position": "early_oct",
                "date": _acc_clock(acc_num).date().isoformat(),
                "hour": _acc_clock(acc_num).hour,
            },
        }
        for acc_num in acc_nums
    ]
    outline = root / f"Authoring/outline/drafts/{PERSONA}/outline.yaml"
    outline.parent.mkdir(parents=True)
    outline.write_text(
        yaml.safe_dump(
            {"meta": {"persona": PERSONA, "status": "complete"}, "sessions": sessions},
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    drafts = root / f"Authoring/sessions/normal/drafts/{PERSONA}"
    drafts.mkdir(parents=True)
    for acc_num in acc_nums:
        if acc_num in forced_accs:
            continue
        (drafts / f"acc_{acc_num:03d}.yaml").write_text(
            NORMAL_SESSION_DRAFT_YAML.replace("user_a", PERSONA).replace(
                "session_id: acc_001", f"session_id: acc_{acc_num:03d}"
            ),
            encoding="utf-8",
        )
    if forced_accs:
        _write_forced_line(root, forced_accs)


def _write_forced_line(root: Path, forced_accs: tuple[int, ...]) -> None:
    """A transcript and draft per forced event."""
    for acc_num in forced_accs:
        event_id = FORCED_EVENT_ID_BY_ACC[acc_num]
        draft = root / f"Authoring/sessions/forced/drafts/{PERSONA}/{event_id}.yaml"
        draft.parent.mkdir(parents=True, exist_ok=True)
        draft.write_text(
            f"event_meta:\n  event_id: {event_id}\n  source_acc_num: {acc_num}\n",
            encoding="utf-8",
        )
        transcript = (
            root
            / f"Authoring/sessions/forced/transcripts/{PERSONA}/{event_id}/transcript.yaml"
        )
        transcript.parent.mkdir(parents=True, exist_ok=True)
        transcript.write_text(
            FORCED_EVENT_TRANSCRIPT_YAML.replace("evolv_05", event_id)
            .replace("user_a", PERSONA)
            .replace("source_acc_num: 75", f"source_acc_num: {acc_num}"),
            encoding="utf-8",
        )


def _write_probe_line(
    root: Path,
    *,
    final_clock: datetime | None = None,
    pre_before_acc: dict[str, int] | None = None,
) -> None:
    final_clock = final_clock or _final_clock()
    pre_before_acc = pre_before_acc or PRE_BEFORE_ACC
    probes = [
        {
            "id": f"fin_{index:02d}",
            "kind": "final",
            "cell": _cell(index),
            "errand": f"w0_errand_{index:02d}",
            "clock": _clock(final_clock),
        }
        for index in range(1, 29)
    ]
    for pre_id, acc_num in pre_before_acc.items():
        fin_index = int(PRE_PAIRS_WITH[pre_id].split("_")[1])
        probes.append(
            {
                "id": pre_id,
                "kind": "pre_event",
                "cell": _cell(fin_index),
                "errand": f"w0_errand_{pre_id}",
                "clock": _clock(_acc_clock(acc_num) - ENVELOPE),
                "pairs_with": PRE_PAIRS_WITH[pre_id],
                "before_acc": acc_num,
            }
        )
    evolving_cells = {
        _cell(int(fin.split("_")[1])): pre_before_acc[pre]
        for pre, fin in PRE_PAIRS_WITH.items()
    }
    cells = {}
    for index in range(1, 29):
        cell = _cell(index)
        if cell in evolving_cells:
            cells[cell] = {
                "pre_setting": "Before",
                "post_setting": "After",
                "event_acc": evolving_cells[cell],
            }
        else:
            cells[cell] = {"setting": "Stable"}
    outline_dir = root / f"Authoring/probe/outline/{PERSONA}"
    outline_dir.mkdir(parents=True)
    (outline_dir / "outline.yaml").write_text(
        yaml.safe_dump(
            {
                "meta": {
                    "persona": PERSONA,
                    "status": "complete",
                    "final": 28,
                    "pre_event": 5,
                    "final_clock": _clock(final_clock),
                },
                # Reverse order on disk: the builder sorts by id.
                "probes": list(reversed(probes)),
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (outline_dir / "resolved_gt.yaml").write_text(
        yaml.safe_dump({"persona": PERSONA, "cells": cells}, sort_keys=False),
        encoding="utf-8",
    )
    paired_finals = {PRE_PAIRS_WITH[pre]: pre for pre in pre_before_acc}
    for probe in probes:
        draft = root / f"Authoring/probe/drafts/{PERSONA}/{probe['id']}/{probe['id']}.yaml"
        draft.parent.mkdir(parents=True)
        # Both halves of a pair declare it, as the real drafts do: the run resolves
        # the overlay from these and the freeze refuses a pairing they do not show.
        meta = {"artifact_kind": "probe_session", "probe_kind": probe["kind"]}
        if probe["kind"] == "pre_event":
            meta["pairs_with"] = PRE_PAIRS_WITH[probe["id"]]
        elif probe["id"] in paired_finals:
            meta["pairs_with"] = paired_finals[probe["id"]]
        draft.write_text(yaml.safe_dump({"meta": meta}, sort_keys=False), encoding="utf-8")


def _pass_forced_transcript_check(monkeypatch: pytest.MonkeyPatch) -> None:
    """D5 checks each forced transcript against its sources at freeze. These
    synthetic transcripts name drafts that are not real sessions, so a
    stand-in passes them; a test that needs the real check restores it."""
    monkeypatch.setattr(
        run_plan_builder, "_validate_forced_transcript", lambda path: None
    )


@pytest.fixture
def synthetic_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    _write_acc_line(tmp_path)
    monkeypatch.setattr(run_plan_builder, "PROJECT_ROOT", tmp_path)
    return tmp_path


def _build(*, script_names: list[str] | None = None, with_probes: bool = True):
    return run_plan_builder._build_frozen_run_plan(
        run_id="run_probe",
        persona_id=PERSONA,
        memory_condition="nanobot_file_memory",
        pa_model="openai/gpt-5.5",
        consolidation_model="openai/gpt-5.5",
        script_names=script_names or [f"acc_{n:03d}" for n in range(1, ACC_COUNT + 1)],
        with_probes=with_probes,
    )


def test_probes_merge_by_cut_point(synthetic_root: Path) -> None:
    _write_probe_line(synthetic_root)
    plan = _build()

    ids = [step.step_id for step in plan.steps]
    assert len(ids) == ACC_COUNT + 33
    # Each pre sits immediately before its before_acc step, in id order.
    for pre_id, acc_num in PRE_BEFORE_ACC.items():
        assert ids[ids.index(f"acc_{acc_num:03d}") - 1] == pre_id
    # The finals follow the last timeline step, sorted by id.
    assert ids[-29] == f"acc_{ACC_COUNT:03d}"
    assert ids[-28:] == [f"fin_{index:02d}" for index in range(1, 29)]
    assert plan.pinned_step_ids() == {
        "acc_009", "acc_029", "acc_049", "acc_069", "acc_089", "acc_099",
    }

    by_id = {step.step_id: step for step in plan.steps}
    for pre_id, acc_num in PRE_BEFORE_ACC.items():
        pre = by_id[pre_id]
        assert pre.kind == "pre_event_probe"
        assert pre.before_acc_num == acc_num
        assert pre.benchmark_session_timestamp == _acc_clock(acc_num) - ENVELOPE
        assert pre.target_setting == "Before"
    for index in range(1, 29):
        final = by_id[f"fin_{index:02d}"]
        assert final.kind == "final_probe"
        assert final.acc_num is None and final.before_acc_num is None
        assert final.benchmark_session_timestamp == _final_clock()
    assert by_id["fin_01"].target_setting == "After"
    assert by_id["fin_02"].target_setting == "Stable"
    for step in plan.steps:
        if step.kind not in {"pre_event_probe", "final_probe"}:
            continue
        assert step.memory_mode == "read_only"
        assert step.stage_policy == "discard"
        assert step.skip_consolidation is True
        assert [step.context, step.target_attribute] == step.target_cell.split("_", 1)
        assert step.script_path == Path(
            f"Authoring/probe/drafts/{PERSONA}/{step.step_id}/{step.step_id}.yaml"
        )
        assert step.probe_overlay is not None
        expected_owner = {
            final: pre for pre, final in PRE_PAIRS_WITH.items()
        }.get(step.step_id, step.step_id)
        assert step.probe_overlay.owner_probe_id == expected_owner
    plan.validate_script_paths(project_root=synthetic_root)


def test_absent_overlay_is_explicit_and_an_old_plan_without_it_loads(
    synthetic_root: Path,
) -> None:
    _write_probe_line(synthetic_root)
    plan = _build()
    step = next(item for item in plan.steps if item.step_id == "fin_02")
    assert step.probe_overlay is not None
    assert step.probe_overlay.owner_probe_id == "fin_02"
    run_dir = synthetic_root / "workspace/runs/legacy_probe"
    run_plan_builder.freeze_resolved_gt_copy(plan, run_dir)
    plan_path = plan.dump_yaml(run_dir / "run_plan.yaml")
    body = yaml.safe_load(plan_path.read_text(encoding="utf-8"))
    for item in body["steps"]:
        if item["kind"] in {"pre_event_probe", "final_probe"}:
            item.pop("probe_overlay", None)
    plan_path.write_text(yaml.safe_dump(body, sort_keys=False), encoding="utf-8")

    assert load_run_plan(plan_path, project_root=synthetic_root).steps


def test_paired_final_ignores_its_unused_fixture_and_binds_pre_owner(
    synthetic_root: Path,
) -> None:
    _write_probe_line(synthetic_root)
    unused = synthetic_root / f"Authoring/probe/drafts/{PERSONA}/fin_01/fixture"
    unused.mkdir()
    (unused / "unused.txt").write_text("not consumed\n", encoding="utf-8")
    plan = _build()
    final = next(item for item in plan.steps if item.step_id == "fin_01")
    pre = next(item for item in plan.steps if item.step_id == "pre_01")
    assert final.probe_overlay == pre.probe_overlay
    assert final.probe_overlay is not None
    assert final.probe_overlay.owner_probe_id == "pre_01"


def test_probe_overlay_rejects_a_foreign_or_wrong_kind_owner(
    synthetic_root: Path,
) -> None:
    _write_probe_line(synthetic_root)
    plan = _build()
    final = next(item for item in plan.steps if item.step_id == "fin_01")
    assert resolve_probe_overlay(plan, final, project_root=synthetic_root) is None

    owner = next(item for item in plan.steps if item.step_id == "pre_01")
    foreign_owner = owner.model_copy(update={"persona_id": "user_y"})
    foreign_plan = plan.model_copy(
        update={
            "steps": [
                foreign_owner if item.step_id == owner.step_id else item
                for item in plan.steps
            ]
        }
    )
    with pytest.raises(RunPlanLoadError, match="different persona"):
        resolve_probe_overlay(
            foreign_plan, final, project_root=synthetic_root
        )

    wrong_kind_owner = owner.model_copy(update={"kind": "final_probe"})
    wrong_kind_plan = plan.model_copy(
        update={
            "steps": [
                wrong_kind_owner if item.step_id == owner.step_id else item
                for item in plan.steps
            ]
        }
    )
    with pytest.raises(RunPlanLoadError, match="wrong probe kind"):
        resolve_probe_overlay(
            wrong_kind_plan, final, project_root=synthetic_root
        )

    final_draft = synthetic_root / final.script_path
    body = yaml.safe_load(final_draft.read_text(encoding="utf-8"))
    body["meta"]["probe_kind"] = "pre_event"
    final_draft.write_text(yaml.safe_dump(body, sort_keys=False), encoding="utf-8")
    with pytest.raises(RunPlanLoadError, match="draft kind does not match"):
        resolve_probe_overlay(plan, final, project_root=synthetic_root)


def test_without_probes_the_plan_is_the_timeline_alone(synthetic_root: Path) -> None:
    plan = _build(with_probes=False)
    assert [step.kind for step in plan.steps] == ["accumulation"] * ACC_COUNT


def test_a_final_off_the_frozen_final_clock_is_refused(synthetic_root: Path) -> None:
    _write_probe_line(synthetic_root)
    outline = synthetic_root / f"Authoring/probe/outline/{PERSONA}/outline.yaml"
    data = yaml.safe_load(outline.read_text(encoding="utf-8"))
    for probe in data["probes"]:
        if probe["id"] == "fin_07":
            probe["clock"]["hour"] += 1
    outline.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="frozen meta.final_clock: id='fin_07'"):
        _build()


# ── an --acc-range shorter than the timeline: only the probes cut inside it ──


def test_a_range_holds_the_pre_probes_whose_cut_point_it_holds(
    synthetic_root: Path,
) -> None:
    """A short run carries the pre probes whose `before_acc` it runs, each in its
    place, and no other probe."""
    _write_probe_line(synthetic_root)

    plan = _build(script_names=[f"acc_{n:03d}" for n in range(1, 31)])

    ids = [step.step_id for step in plan.steps]
    assert [i for i in ids if not i.startswith("acc_")] == ["pre_01", "pre_02"]
    assert ids[ids.index("acc_010") - 1] == "pre_01"
    assert ids[ids.index("acc_030") - 1] == "pre_02"
    assert plan.pinned_step_ids() == {"acc_009", "acc_029"}


def test_a_range_holding_no_cut_point_freezes_no_probe_steps(
    synthetic_root: Path,
) -> None:
    _write_probe_line(synthetic_root)

    plan = _build(script_names=[f"acc_{n:03d}" for n in range(1, 6)])

    assert [step.step_id for step in plan.steps] == [
        f"acc_{n:03d}" for n in range(1, 6)
    ]


def test_the_finals_come_in_only_with_the_timeline_end(synthetic_root: Path) -> None:
    """The finals' cut point is after the Outline's last session (acc_099 on
    this line): a range that runs it carries all 28 after it, one that stops
    short carries none."""
    _write_probe_line(synthetic_root)

    to_the_end = _build(script_names=[f"acc_{n:03d}" for n in range(95, 100)])
    ids = [step.step_id for step in to_the_end.steps]
    assert ids == [f"acc_{n:03d}" for n in range(95, 100)] + [
        f"fin_{index:02d}" for index in range(1, 29)
    ]
    assert to_the_end.pinned_step_ids() == {"acc_099"}

    short = _build(script_names=[f"acc_{n:03d}" for n in range(95, 99)])
    assert [step.kind for step in short.steps] == ["accumulation"] * 4


def test_a_range_starting_at_a_cut_point_has_nothing_to_branch_from(
    synthetic_root: Path,
) -> None:
    """`before_acc` 10 is in a range that starts at 10, so pre_01 comes in, but
    no timeline step runs before it: the plan refuses rather than branch a
    probe off nothing."""
    _write_probe_line(synthetic_root)
    with pytest.raises(
        ValueError,
        match="a probe must follow an accumulation/forced_event step.*'pre_01'",
    ):
        _build(script_names=[f"acc_{n:03d}" for n in range(10, 13)])


def test_setting_for_reads_the_record_by_kind() -> None:
    assert setting_for("final", {"setting": "S"}) == "S"
    assert setting_for("pre_event", {"setting": "S"}) == "S"
    evolving = {"pre_setting": "P", "post_setting": "Q", "event_acc": 15}
    assert setting_for("pre_event", evolving) == "P"
    assert setting_for("final", evolving) == "Q"
    with pytest.raises(ValueError, match="GT record must be"):
        setting_for("final", {"pre_setting": "P"})
    with pytest.raises(ValueError, match="GT record must be"):
        setting_for("final", {"setting": "S", "post_setting": "Q"})
    with pytest.raises(ValueError, match="probe kind must be one of"):
        setting_for("post", {"setting": "S"})


# ── the frozen resolved_gt copy ──────────────────────────


def test_with_probes_records_the_frozen_resolved_gt(synthetic_root: Path) -> None:
    _write_probe_line(synthetic_root)
    plan = _build()

    assert plan.resolved_gt is not None
    assert plan.resolved_gt.path == "resolved_gt.yaml"
    assert _build(with_probes=False).resolved_gt is None


def test_freeze_resolved_gt_copy_writes_the_recorded_bytes(synthetic_root: Path) -> None:
    _write_probe_line(synthetic_root)
    source = synthetic_root / f"Authoring/probe/outline/{PERSONA}/resolved_gt.yaml"
    plan = _build()
    run_dir = synthetic_root / "workspace/runs/run_probe"

    target = run_plan_builder.freeze_resolved_gt_copy(plan, run_dir)

    assert target == run_dir / "resolved_gt.yaml"
    assert target.read_bytes() == source.read_bytes()
    # The plan then loads back against its own copy.
    plan_path = plan.dump_yaml(run_dir / "run_plan.yaml")
    assert load_run_plan(plan_path, project_root=synthetic_root).resolved_gt == plan.resolved_gt


def test_freeze_resolved_gt_copy_copies_the_file_as_it_is_now(synthetic_root: Path) -> None:
    _write_probe_line(synthetic_root)
    source = synthetic_root / f"Authoring/probe/outline/{PERSONA}/resolved_gt.yaml"
    plan = _build()
    source.write_text(source.read_text(encoding="utf-8") + "# edited\n", encoding="utf-8")
    run_dir = synthetic_root / "workspace/runs/run_probe"

    target = run_plan_builder.freeze_resolved_gt_copy(plan, run_dir)
    assert target.read_bytes() == source.read_bytes()

    with pytest.raises(ValueError, match="records no resolved_gt"):
        run_plan_builder.freeze_resolved_gt_copy(_build(with_probes=False), run_dir)


def test_persona_has_probe_line_needs_both_files(synthetic_root: Path) -> None:
    assert run_plan_builder.persona_has_probe_line(PERSONA) is False
    _write_probe_line(synthetic_root)
    assert run_plan_builder.persona_has_probe_line(PERSONA) is True
    (synthetic_root / f"Authoring/probe/outline/{PERSONA}/resolved_gt.yaml").unlink()
    with pytest.raises(FileNotFoundError, match="outline=True resolved_gt=False"):
        run_plan_builder.persona_has_probe_line(PERSONA)


def test_load_script_routes_a_probe_draft_through_the_authored_loader(tmp_path: Path) -> None:
    from test_probe_session_authoring import _probe

    path = tmp_path / "fin_02.yaml"
    path.write_text(yaml.safe_dump(_probe()), encoding="utf-8")

    script = run_plan_builder._load_script(path)

    assert script.artifact_kind == "probe_session"
    assert script.session_id == "fin_02"
    assert script.session_type == "final"


# ── the full 1..108 skeleton: 99 authored sessions and 9 gap days ──


@pytest.fixture
def full_skeleton_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A persona's real shape: 108 acc slots, 99 of them authored (5 forced
    events among them), the other 9 gap days the Outline never wrote."""
    authored = tuple(
        acc_num
        for acc_num in range(1, FULL_TIMELINE_ACC_COUNT + 1)
        if acc_num not in GAP_ACCS
    )
    _write_acc_line(tmp_path, acc_nums=authored, forced_accs=FORCED_ACCS)
    # The pre-event probes cut in front of the forced events, and the finals
    # one envelope after the last authored session.
    _write_probe_line(
        tmp_path,
        final_clock=_acc_clock(FULL_TIMELINE_ACC_COUNT) + ENVELOPE,
        pre_before_acc={
            f"pre_{index:02d}": acc_num
            for index, acc_num in enumerate(FORCED_ACCS, start=1)
        },
    )
    monkeypatch.setattr(run_plan_builder, "PROJECT_ROOT", tmp_path)
    _pass_forced_transcript_check(monkeypatch)
    return tmp_path


def test_a_full_range_plan_fills_its_gap_days_and_has_the_full_shape(
    full_skeleton_root: Path,
) -> None:
    plan = _build(
        script_names=[
            f"acc_{n:03d}" for n in range(1, FULL_TIMELINE_ACC_COUNT + 1)
        ]
    )

    timeline = [
        step for step in plan.steps if step.kind in {"accumulation", "forced_event"}
    ]
    assert [step.acc_num for step in timeline] == list(
        range(1, FULL_TIMELINE_ACC_COUNT + 1)
    )
    placeholders = [step for step in timeline if step.placeholder]
    assert [step.acc_num for step in placeholders] == list(GAP_ACCS)
    assert len(timeline) - len(placeholders) == 99
    # The builder ran the shape check itself; running it again is the receipt.
    validate_full_timeline_shape(plan)

    # A gap day reads nothing off disk and is never a cut point.
    gap = placeholders[0]
    assert not (full_skeleton_root / gap.script_path).exists()
    plan.validate_script_paths(project_root=full_skeleton_root)
    assert not {step.step_id for step in placeholders} & plan.pinned_step_ids()
    # It holds the slot without spending an envelope: the clock is the
    # previous authored session's.
    before = timeline[timeline.index(gap) - 1]
    assert gap.benchmark_session_timestamp == before.benchmark_session_timestamp


def test_the_full_timeline_carries_every_probe_where_it_always_stood(
    full_skeleton_root: Path,
) -> None:
    """Scoping probes to the requested accs leaves the full plan as it was:
    all 33 probes, each pre right before its forced event, the finals last,
    132 steps besides the 9 gap days."""
    plan = _build(
        script_names=[
            f"acc_{n:03d}" for n in range(1, FULL_TIMELINE_ACC_COUNT + 1)
        ]
    )

    pre_by_acc = {
        acc_num: f"pre_{index:02d}"
        for index, acc_num in enumerate(FORCED_ACCS, start=1)
    }
    expected: list[str] = []
    for acc_num in range(1, FULL_TIMELINE_ACC_COUNT + 1):
        if acc_num in pre_by_acc:
            expected += [pre_by_acc[acc_num], FORCED_EVENT_ID_BY_ACC[acc_num]]
        else:
            expected.append(f"acc_{acc_num:03d}")
    expected += [f"fin_{index:02d}" for index in range(1, 29)]
    assert [step.step_id for step in plan.steps] == expected
    assert sum(not step.placeholder for step in plan.steps) == 132


def test_a_sub_range_is_not_held_to_the_full_shape(full_skeleton_root: Path) -> None:
    """`--acc-range` is a slice of the run, so it meets none of the counts."""
    plan = _build(
        script_names=[f"acc_{n:03d}" for n in range(1, 21)], with_probes=False
    )

    assert [step.acc_num for step in plan.steps if step.acc_num] == list(range(1, 21))
    assert [step.acc_num for step in plan.steps if step.placeholder] == [16]


def test_an_acc_beyond_the_skeleton_is_still_a_missing_session(
    full_skeleton_root: Path,
) -> None:
    with pytest.raises(
        ValueError,
        match="missing from the Authoring Outline: script_name='acc_109'",
    ):
        _build(script_names=["acc_109"], with_probes=False)


# ── a frozen plan reads its drafts as they are ────────────────────


def test_a_probe_draft_edited_after_the_freeze_still_loads(
    synthetic_root: Path,
) -> None:
    """A plan reloads at every resume and branch; nothing ties it to the
    draft's bytes at freeze."""
    _write_probe_line(synthetic_root)
    plan = _build()
    plan_path = plan.dump_yaml(synthetic_root / "workspace/runs/run_probe/run_plan.yaml")
    run_plan_builder.freeze_resolved_gt_copy(
        plan, synthetic_root / "workspace/runs/run_probe"
    )
    assert load_run_plan(plan_path, project_root=synthetic_root).steps

    draft = synthetic_root / f"Authoring/probe/drafts/{PERSONA}/fin_03/fin_03.yaml"
    draft.write_text(draft.read_text(encoding="utf-8") + "# edited\n", encoding="utf-8")

    assert load_run_plan(plan_path, project_root=synthetic_root).steps
