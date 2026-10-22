"""T22 — the contract-12 version boundary, on the artefacts a run produces.

A frozen `RunPlan` records its
`ix_contract_version` and identity triple; nothing refuses a plan on them.
What is here is (b) and (c): what an old `ix_render` record does on the way
into a judge pack, and whether the records a live turn produces really carry
the versions the plan froze.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from harness.judge_pack import build_text_judge_pack
from harness.run_plan import RunPlan
from harness.session_executor import _ix_render_versions
from harness.transcript import TranscriptWriter, load_events, read_ix_render
from pa.agent.ix.render_types import (
    IX_CONTRACT_VERSION,
    IX_RENDER_SCHEMA_VERSION,
    IX_RENDER_OPTIONAL_KEYS,
    IX_RENDER_TOP_LEVEL_KEYS,
    IXContractVersionError,
    from_ix_render_metadata,
)
from pa.providers.base import LLMResponse

from .conftest import (
    ScriptProvider,
    plan_call,
    run,
    run_turn,
    selection,
    task_call,
)


def _pa_turn(ix_render: dict | None) -> dict:
    event: dict[str, Any] = {
        "t": 1,
        "kind": "pa_turn",
        "beat": "open",
        "outbound": "I read your note. Shall I send it?",
        "tool_events": [{"name": "mcp_state_document_read", "status": "ok", "args": {}}],
    }
    if ix_render is not None:
        event["ix_render"] = ix_render
    return event


def _current_blob(**overrides: Any) -> dict:
    blob = {
        "schema_version": IX_RENDER_SCHEMA_VERSION,
        "ix_contract_version": IX_CONTRACT_VERSION,
        "episodes": {},
        "released": {"batches": [], "questions": [], "plan_executed": []},
        "held": {
            "batches": [], "questions": [], "plan_remaining": [], "calls": [],
            "cancelled": [], "outreach_message": "",
        },
        "gate": None,
        "consent": [],
        "notices": [],
        "completion": {"released": False, "text": ""},
        "reasoning": {
            "shown": False,
            "source": "user_facing_rationale",
            "provider_reasoning_present": False,
        },
        "certainties": [],
        "proactive_decision": None,
        "trace": [],
    }
    blob.update(overrides)
    return blob


# ---------------------------------------------------------------------------
# T22 (b) · an old record is refused, never reinterpreted
# ---------------------------------------------------------------------------

def test_an_ix_render_without_a_schema_version_is_refused() -> None:
    """T22 (b): a blob missing the version key cannot be read as a new one."""
    blob = _current_blob()
    del blob["schema_version"]

    with pytest.raises(ValueError, match="missing keys: schema_version"):
        read_ix_render(_pa_turn(blob))


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", "ix_render/0.9.0"),
        ("ix_contract_version", "ix-wrapper/0.9.0"),
    ],
)
def test_a_stale_version_raises_rather_than_being_reinterpreted(
    field: str, value: str
) -> None:
    """T22 (b): a version boundary fails fast on both halves of the stamp."""
    with pytest.raises(IXContractVersionError, match="0.9.0"):
        read_ix_render(_pa_turn(_current_blob(**{field: value})))


def test_an_unknown_key_in_the_blob_is_refused_too() -> None:
    """A record from a newer writer is not silently narrowed to today's keys."""
    with pytest.raises(ValueError, match="unknown keys: future_field"):
        from_ix_render_metadata(_current_blob(future_field=1))


def test_a_transcript_written_before_the_wrapper_takes_the_explicit_legacy_path(
    tmp_path: Path,
) -> None:
    """T22 (b), the other half: no `ix_render` at all is a legacy record.

    Absence is not a version mismatch — it is a pre-wrapper transcript. The
    reader says "nothing here" rather than inventing a blob, the step records
    no version stamps, and the text pack still builds from what the record has.
    """
    run_dir = tmp_path / "legacy"
    with TranscriptWriter(run_dir) as writer:
        writer.log_event({"t": 0, "kind": "simulator_turn", "message": "Read my note."})
        writer.log_event(_pa_turn(None))

    events = load_events(run_dir / "transcript.jsonl")

    assert read_ix_render(events[1]) is None
    assert _ix_render_versions(events) == {}
    pack = build_text_judge_pack(events)
    assert [turn["kind"] for turn in pack["turns"]] == ["user_message", "pa_turn"]
    assert "ix_render" not in json.dumps(pack)


def test_a_current_record_reads_back_with_exactly_the_frozen_key_set() -> None:
    blob = read_ix_render(_pa_turn(_current_blob()))

    assert blob is not None
    # `raw_blocks` is optional: this turn wrote no carrier block.
    assert list(blob) == [
        key for key in IX_RENDER_TOP_LEVEL_KEYS if key not in IX_RENDER_OPTIONAL_KEYS
    ]
    assert blob["schema_version"] == IX_RENDER_SCHEMA_VERSION
    assert blob["ix_contract_version"] == IX_CONTRACT_VERSION


# ---------------------------------------------------------------------------
# T22 (c) · what a live turn stamps, and what the plan froze
# ---------------------------------------------------------------------------

def test_a_rendered_turn_stamps_both_versions_and_they_survive_the_transcript(
    loop_factory, tmp_path: Path
) -> None:
    """T22 (c): the produced record carries both stamps, non-empty, and reads back."""
    async def scenario() -> None:
        provider = ScriptProvider([
            LLMResponse(content="Checking.", tool_calls=[
                selection("autonomy_level", "Reactive"),
                plan_call({"id": "a1", "bound_item_ids": ["i1"]}),
                task_call("send_1", "task_send", "i1", ask="Shall I send it?"),
            ]),
        ])
        loop, _ = loop_factory(provider, "task_send")

        outbound = await run_turn(
            loop, "Mail Dale.", active_ix=["autonomy_level"],
            action_types={"task_send": "external_action"},
        )

        blob = outbound.metadata["ix_render"]
        assert blob["schema_version"] == IX_RENDER_SCHEMA_VERSION
        assert blob["ix_contract_version"] == IX_CONTRACT_VERSION
        assert blob["schema_version"] and blob["ix_contract_version"]

        run_dir = tmp_path / "live"
        with TranscriptWriter(run_dir) as writer:
            writer.log_event({
                "t": 0, "kind": "pa_turn", "beat": "open",
                "outbound": outbound.content,
                "tool_events": outbound.metadata["tool_events"],
                "ix_render": blob,
            })
        events = load_events(run_dir / "transcript.jsonl")

        assert _ix_render_versions(events) == {
            "ix_render_schema_version": IX_RENDER_SCHEMA_VERSION,
            "ix_contract_version": IX_CONTRACT_VERSION,
        }

    run(scenario())


def test_the_version_a_plan_freezes_is_the_one_records_carry() -> None:
    """T22 (c): plan and record read the same constant.

    If these two ever came from different places, a run could produce records
    the plan does not describe — which is the whole failure mode contract 12
    exists to make impossible.
    """
    frozen = RunPlan.model_fields["ix_contract_version"].default_factory()

    assert frozen == IX_CONTRACT_VERSION
    assert _current_blob()["ix_contract_version"] == frozen
