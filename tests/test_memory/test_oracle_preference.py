"""Tests for the oracle_preference controlled condition provider."""

from __future__ import annotations

import re
from datetime import datetime, timezone

import pytest

from memory.adapter import MemoryAdapter
from memory.conditions.oracle_preference import OraclePreferenceAdapter
from memory.events import MemoryEvent
from memory.results import WriteStatus
from memory.scope import MemoryRetrievalBudget, MemoryScope, RetrieveRequest
from pa.agent.ix.settings_table import load_setting_definitions
from scripts.resolved_gt import setting_for


def _request() -> RetrieveRequest:
    return RetrieveRequest(
        query="anything",
        scope=MemoryScope(run_id="run_001", persona_id="user_b"),
        session_id="s01",
        budget=MemoryRetrievalBudget(max_tokens=4000),
    )


def _instruction(attribute: str, setting: str) -> str:
    row = {row.id: row for row in load_setting_definitions().attributes}[attribute]
    return {option.name: option.pa_instruction for option in row.settings}[setting]


#: An evolving cell as `resolved_gt.yaml` records it (user_b, work_tone_formality).
EVOLVING_RECORD = {
    "pre_setting": "Casual",
    "post_setting": "Consultative",
    "event_acc": 15,
}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("kind", "setting"), [("pre_event", "Casual"), ("final", "Consultative")]
)
async def test_retrieve_is_the_resolved_ground_truth_and_its_definition(
    kind: str, setting: str
) -> None:
    gt_setting = setting_for(kind, EVOLVING_RECORD)
    assert gt_setting == setting
    adapter = OraclePreferenceAdapter("work_tone_formality", gt_setting)

    result = await adapter.retrieve(_request())

    assert len(result.raw) == 1
    assert result.trace is not None and result.trace.retrieved_count == 1
    assert adapter.format_context(result) == (
        "In work situations like this one, the user prefers Tone & Formality: "
        f"{setting}. {_instruction('tone_formality', setting)}"
    )


@pytest.mark.asyncio
async def test_wording_never_names_the_measurement() -> None:
    """Every cell the taxonomy has, in both contexts."""
    for row in load_setting_definitions().attributes:
        for option in row.settings:
            for context in ("work", "personal"):
                adapter = OraclePreferenceAdapter(f"{context}_{row.id}", option.name)
                text = adapter.format_context(await adapter.retrieve(_request()))
                assert text.startswith(f"In {context} situations like this one, ")
                assert not re.search(
                    r"benchmark|probe|\btest|evaluat|oracle|ground truth",
                    text,
                    re.IGNORECASE,
                ), text


@pytest.mark.asyncio
async def test_writes_are_no_ops_like_no_memory() -> None:
    adapter = OraclePreferenceAdapter("work_verbosity", "Terse")
    assert isinstance(adapter, MemoryAdapter)
    receipt = await adapter.record_event(
        MemoryEvent(
            event_id="evt_1",
            run_id="run_001",
            persona_id="user_b",
            agent_id="nanobot",
            session_id="s01",
            turn_id="t01",
            timestamp=datetime.now(timezone.utc),
            messages=[{"role": "user", "content": "I like long answers"}],
        )
    )
    assert receipt.status == WriteStatus.skipped_no_memory
    assert receipt.trace is not None
    assert receipt.trace.backend_name == "oracle_preference"


@pytest.mark.asyncio
async def test_without_a_target_it_refuses_to_retrieve() -> None:
    """The entrypoint's run-level adapter: only a probe branch knows its cell."""
    adapter = OraclePreferenceAdapter()
    with pytest.raises(RuntimeError, match="outside a probe branch"):
        await adapter.retrieve(_request())


def test_a_setting_the_attribute_does_not_have_is_refused() -> None:
    with pytest.raises(KeyError, match="Consultative"):
        OraclePreferenceAdapter("work_verbosity", "Consultative")
