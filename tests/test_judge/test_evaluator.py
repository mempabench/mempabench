"""LLM judge runner — prompt assembly, strict reply schema, evidence re-ask.

No network: the client is injected. Run:
    python -m pytest tests/test_judge/test_evaluator.py -q
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from data.personas.coverage import TAXONOMY
from judge.evaluator import (
    BLOCK_SCENE,
    BLOCK_TOOL_TRAJECTORY,
    FEATURES_DIR,
    OMITTED_NOTE,
    UNUSED_BLOCKS,
    JudgeOutputInvalid,
    JudgeReply,
    assemble_prompt,
    judge_id_for,
    load_judge_input,
    main,
    omitted_blocks,
    parse_reply,
    rubric_markdown,
    run_judge,
    split_model_id,
)

SNAPSHOT = Path(__file__).parent / "snapshots" / "verbosity_prompt.md"
MODEL = "google/gemini-3.8-flash"
PA_REPLY = (
    "Software under $200 goes through the departmental filing route; you file the receipt "
    "within 30 days. Note the policy excludes subscriptions renewed monthly."
)


def write_judge_input(run_dir: Path, probe_id: str = "fin_02", attribute: str = "verbosity") -> Path:
    setting = TAXONOMY[attribute][-1]
    probe = run_dir / "judge_input" / probe_id
    probe.mkdir(parents=True)
    (run_dir / "judge_input" / "index.json").write_text(
        json.dumps({"run_id": "run_x", "probes": [probe_id]}), encoding="utf-8"
    )
    (probe / "meta.json").write_text(
        json.dumps({"run_id": "run_x", "step_id": probe_id, "kind": "final_probe", "context": "work"}),
        encoding="utf-8",
    )
    (probe / "text_judge_pack.json").write_text(
        json.dumps(
            {
                "pack_schema_version": "text_judge_pack/1.0.0",
                "meta": {"run_id": "run_x", "kind": "final_probe"},
                "turns": [
                    {"t": 3, "kind": "user_message", "beat": "beat1",
                     "text": "Can I expense the $120 editor license? What's the route?"},
                    {"t": 5, "kind": "pa_turn", "beat": "beat1", "public_turn": PA_REPLY,
                     "task_tool_events": [
                         {"order": 0, "name": "document_read", "status": "ok",
                          "args": {"path": "documents/admin/software_reimbursement_policy.md"}}
                     ]},
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (probe / "tool_log.jsonl").write_text(
        json.dumps({"t": 1, "tool": "document_read", "status": "ok",
                    "args": {"path": "documents/admin/software_reimbursement_policy.md"},
                    "result_summary": {"path": "documents/admin/software_reimbursement_policy.md",
                                       "bytes": 1180}}) + "\n",
        encoding="utf-8",
    )
    (probe / "ground_truth.json").write_text(
        json.dumps({
            "persona_id": "user_a", "context": "work", "attribute": attribute,
            "setting": setting, "confidence": "High", "source": "extractor_high",
            "source_path": "data/personas/user_a_preferences.yaml",
            "allowed_settings": list(TAXONOMY[attribute]),
            "evolving": {"is_evolving": False, "pre_setting": None, "post_setting": None},
        }),
        encoding="utf-8",
    )
    (probe / "scene.json").write_text(
        json.dumps({
            "probe_id": probe_id,
            "core_task": "Tell the user whether the $120 editor license is reimbursable and how to file it.",
            "triggers": [{"trigger_id": "policy_update", "background": "Policy revised last month.",
                          "detail": "Cap moved to $200.", "user_impact": "Filing the old way gets rejected."}],
            "world_state": {"feasible_options": [], "pending_items": []},
            "familiarity_signals": None,
        }),
        encoding="utf-8",
    )
    (probe / "ix_selection.json").write_text(
        json.dumps({"probe_id": probe_id, "target_attribute": attribute, "context": "work",
                    "selections": [{"beat": "beat1", "call_index": 0, "attribute": attribute,
                                    "setting": setting, "evidence": "e", "application": "a"}],
                    "selected_setting": setting, "selection_count": 1, "no_selection": False}),
        encoding="utf-8",
    )
    return probe


class FakeClient:
    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.calls: list[tuple[str, str]] = []

    def complete(self, *, system: str, user: str) -> JudgeReply:
        self.calls.append((system, user))
        return JudgeReply(content=self.replies.pop(0), model=MODEL,
                          prompt_tokens=100, completion_tokens=20, usd=0.001)


def valid_reply(**overrides) -> str:
    reply = {
        "score": 4,
        "rationale": "Answer first, then two support classes: boundary condition and caveat.",
        "evidence": [
            {"type": "quote", "text": "Note the policy excludes subscriptions renewed monthly."},
            {"type": "tool_event", "t": 1, "tool": "document_read"},
        ],
        "asked_user_for_preference": False,
        "flags": {"content_error_corrected_by_simulator": False, "task_failed": False,
                  "probe_degenerate": False},
    }
    reply.update(overrides)
    return json.dumps(reply)


# ------------------------------------------------------------------ prompt files


def test_every_taxonomy_attribute_has_a_feature_table() -> None:
    assert {p.stem for p in FEATURES_DIR.glob("*.md")} == set(TAXONOMY)
    for attribute in TAXONOMY:
        text = rubric_markdown(attribute)
        assert text.startswith(f"**属性** `{attribute}`")
        assert "| Setting |" in text


def test_process_visibility_rubric_counts_the_actual_tool_log() -> None:
    # D4 (owner ruling 2026-09-16): the authoring side plans a readback call after
    # every write, so an outline counts three calls where the transcript may hold
    # two. The judge must count what `tool_log` really contains.
    text = rubric_markdown("process_visibility")
    assert "**N 就是 `tool_log` 里实际出现的 task 调用条数。**" in text
    assert "judge **不得假设 readback 真的发生了**" in text


def test_prompt_assembly_snapshot_for_verbosity(tmp_path: Path, update_snapshots: bool) -> None:
    system, user = assemble_prompt(load_judge_input(write_judge_input(tmp_path)))
    rendered = f"### SYSTEM\n\n{system}\n\n### USER\n\n{user}\n"
    if update_snapshots:
        SNAPSHOT.write_text(rendered, encoding="utf-8")
    assert rendered == SNAPSHOT.read_text(encoding="utf-8")


def test_seven_blocks_put_evidence_before_criteria(tmp_path: Path) -> None:
    system, user = assemble_prompt(load_judge_input(write_judge_input(tmp_path)))
    assert "| **5** |" in system and "直接向用户索要这项偏好" in system
    order = ["# 本条 probe", "# 场景", "# 对话记录", "# 工具轨迹", "# 本属性 rubric", "# 输出指令"]
    positions = [user.index(head) for head in order]
    assert positions == sorted(positions)
    assert "ground-truth setting: **`Detailed`**" in user
    assert "context: `work`" in user
    assert PA_REPLY in user
    assert "## 2. `verbosity`" in user
    # The text face never sees the IX selection.
    assert "ix_selection" not in user and "selected_setting" not in user


# ------------------------------------------------- per-attribute block drops


def test_unused_blocks_covers_the_taxonomy_and_only_droppable_blocks() -> None:
    assert set(UNUSED_BLOCKS) == set(TAXONOMY)
    droppable = {BLOCK_SCENE, BLOCK_TOOL_TRAJECTORY}
    assert all(blocks <= droppable for blocks in UNUSED_BLOCKS.values())


def test_attribute_that_drops_the_tool_trajectory(tmp_path: Path) -> None:
    """`uncertainty_expression` row, 用不到的文件: tool_log、state_diff、pa_toolcalls、scene."""
    probe = write_judge_input(tmp_path, attribute="uncertainty_expression")
    _, user = assemble_prompt(load_judge_input(probe))
    assert omitted_blocks("uncertainty_expression") == (BLOCK_SCENE, BLOCK_TOOL_TRAJECTORY)
    assert OMITTED_NOTE[BLOCK_SCENE] in user and OMITTED_NOTE[BLOCK_TOOL_TRAJECTORY] in user
    assert "t=1 tool=`document_read`" not in user
    assert "Filing the old way gets rejected." not in user
    # The seven blocks keep their headings and their order even when two are dropped.
    order = ["# 本条 probe", "# 场景", "# 对话记录", "# 工具轨迹", "# 本属性 rubric", "# 输出指令"]
    positions = [user.index(head) for head in order]
    assert positions == sorted(positions)
    # probe_info / transcript / rubric / output are never dropped.
    assert "目标属性: `uncertainty_expression`" in user
    assert PA_REPLY in user
    assert "## 6. `uncertainty_expression`" in user and '"score"' in user


def test_attribute_that_keeps_every_block(tmp_path: Path) -> None:
    """`task_expansion` row, 用不到的文件: pa_toolcalls（仅备查）— no prompt block."""
    probe = write_judge_input(tmp_path, attribute="task_expansion")
    _, user = assemble_prompt(load_judge_input(probe))
    assert omitted_blocks("task_expansion") == ()
    assert "`policy_update`" in user and "Filing the old way gets rejected." in user
    assert "t=1 tool=`document_read`" in user
    assert OMITTED_NOTE[BLOCK_SCENE] not in user
    assert OMITTED_NOTE[BLOCK_TOOL_TRAJECTORY] not in user


# -------------------------------------------------------------------- reply shape


@pytest.mark.parametrize(
    "bad",
    [
        "not json",
        json.dumps({"score": 4}),
        valid_reply(score=6),
        valid_reply(score="4"),
        valid_reply(evidence=[]),
        valid_reply(evidence=[{"type": "tool_log", "t": 1, "tool": "x"}]),
        valid_reply(evidence=[{"type": "quote", "text": "x", "extra": 1}]),
        valid_reply(asked_user_for_preference="no"),
        valid_reply(flags={"task_failed": False}),
    ],
)
def test_parse_reply_rejects_off_schema_output(bad: str) -> None:
    with pytest.raises(JudgeOutputInvalid):
        parse_reply(bad)


def test_parse_reply_accepts_fenced_json() -> None:
    assert parse_reply("```json\n" + valid_reply() + "\n```")["score"] == 4


# --------------------------------------------------------------------------- run


def test_valid_reply_writes_the_record(tmp_path: Path) -> None:
    probe = write_judge_input(tmp_path)
    client = FakeClient([valid_reply()])
    out = run_judge(probe, client, model=MODEL)
    assert out == tmp_path / "judge_output" / "llm__google_gemini-3.8-flash" / "fin_02.json"
    record = json.loads(out.read_text(encoding="utf-8"))
    assert record["schema_version"] == 1
    assert record["probe_id"] == "fin_02" and record["run_id"] == "run_x"
    assert record["judge_id"] == judge_id_for(MODEL) and record["judge_kind"] == "llm"
    assert record["ix_face"] == {"gt_setting": "Detailed", "pa_setting": "Detailed",
                                 "exact_match": True, "no_selection": False, "computed_by": "code"}
    assert record["text_face"]["score"] == 4
    assert record["text_face"]["score_domain"] == [1, 2, 3, 4, 5]
    assert record["text_face"]["asked_user_for_preference"] is False
    assert record["flags"]["task_failed"] is False
    assert record["cost"] == {"model": MODEL, "prompt_tokens": 100, "completion_tokens": 20, "usd": 0.001}
    assert len(client.calls) == 1


def test_probe_degenerate_flag_reaches_the_record(tmp_path: Path) -> None:
    """system.md 与 autonomy_level.md 都指示 judge 在材料缺陷时置这个 flag——它必须落进记录。"""
    probe = write_judge_input(tmp_path)
    flags = {"content_error_corrected_by_simulator": False, "task_failed": False,
             "probe_degenerate": True}
    client = FakeClient([valid_reply(flags=flags)])
    record = json.loads(run_judge(probe, client, model=MODEL).read_text(encoding="utf-8"))
    assert record["flags"]["probe_degenerate"] is True


def test_rejected_evidence_is_reasked_once_then_accepted(tmp_path: Path) -> None:
    probe = write_judge_input(tmp_path)
    bad = valid_reply(evidence=[{"type": "quote", "text": "this sentence was never said"}])
    client = FakeClient([bad, valid_reply()])
    record = json.loads(run_judge(probe, client, model=MODEL).read_text(encoding="utf-8"))
    assert len(client.calls) == 2
    assert "上一次输出被拒绝" in client.calls[1][1]
    assert "never said" in client.calls[1][1]
    assert record["cost"]["prompt_tokens"] == 200 and record["cost"]["usd"] == pytest.approx(0.002)


def test_invalid_twice_raises_and_writes_nothing(tmp_path: Path) -> None:
    probe = write_judge_input(tmp_path)
    client = FakeClient(["garbage", valid_reply(evidence=[{"type": "tool_event", "t": 9, "tool": "x"}])])
    with pytest.raises(JudgeOutputInvalid, match="rejected twice"):
        run_judge(probe, client, model=MODEL)
    assert not (tmp_path / "judge_output").exists()
    assert len(client.calls) == 2


def test_model_id_must_be_provider_slash_model() -> None:
    assert split_model_id("z-ai/glm-5.2:free") == ("z-ai", "glm-5.2:free")
    for bad in ("gemini", "a/b/c", "/x", "x/"):
        with pytest.raises(ValueError):
            split_model_id(bad)


def test_dry_run_prints_the_prompt_and_calls_nothing(tmp_path: Path, capsys, monkeypatch) -> None:
    probe = write_judge_input(tmp_path)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    assert main(["--judge-input", str(probe), "--model", MODEL, "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "### judge_id: llm__google_gemini-3.8-flash" in out
    assert "### SYSTEM" in out and "### USER" in out and PA_REPLY in out
    assert "### omitted blocks (§4.4): scene, tool_trajectory" in out
    assert not (tmp_path / "judge_output").exists()


def test_dry_run_reports_when_no_block_is_omitted(tmp_path: Path, capsys, monkeypatch) -> None:
    probe = write_judge_input(tmp_path, attribute="task_expansion")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    assert main(["--judge-input", str(probe), "--model", MODEL, "--dry-run"]) == 0
    assert "### omitted blocks (§4.4): （无）" in capsys.readouterr().out
