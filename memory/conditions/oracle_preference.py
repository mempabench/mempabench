"""Oracle-preference controlled condition — the probe's true setting as memory.

A realization ceiling, not a memory architecture (Obsidian
`Plans/Oracle_Preference_Condition_20260925.md`). Writes are no-ops exactly as
in `NoMemoryAdapter`. `retrieve()` returns one entry: the true setting of the
probe's target cell plus that setting's `pa_instruction` from
`configs/setting_definitions.yaml`, which the PA reads in its usual
`# Retrieved Memory` slot. The setting is the probe step's frozen
`target_setting` — `resolved_gt.yaml` read through `setting_for`
(`harness/run_plan_builder.py::_probe_steps`) — and the orchestrator builds one
adapter per probe branch (`harness/orchestrator.py::_run_probe_in_branch`).
"""

from __future__ import annotations

from memory.conditions.no_memory import NoMemoryAdapter
from memory.results import ConsistencyModel, RetrieveResult, Trace
from memory.scope import RetrieveRequest
from pa.agent.ix.settings_table import load_setting_definitions


class OraclePreferenceAdapter(NoMemoryAdapter):
    """Controlled condition: no memory, except the probe cell's true setting.

    Built without a target by the run entrypoint, like every other condition;
    only a probe branch knows its cell, so only a branch adapter can retrieve.
    """

    backend_name: str = "oracle_preference"

    def __init__(
        self, target_cell: str | None = None, setting: str | None = None
    ) -> None:
        super().__init__()
        self._text: str | None = None
        if target_cell is not None:
            context, attribute = target_cell.split("_", 1)
            row = {
                row.id: row for row in load_setting_definitions().attributes
            }[attribute]
            instruction = {
                option.name: option.pa_instruction for option in row.settings
            }[setting]
            self._text = (
                f"In {context} situations like this one, the user prefers "
                f"{row.label}: {setting}. {instruction}"
            )

    async def retrieve(self, request: RetrieveRequest) -> RetrieveResult:
        if self._text is None:
            raise RuntimeError(
                "oracle_preference has no target cell outside a probe branch, "
                "so it has nothing to retrieve"
            )
        return RetrieveResult(
            raw=[{"memory": self._text}],
            trace=Trace(
                backend_name=self.backend_name,
                scope=request.scope,
                native_operation="retrieve",
                native_ids=[],
                latency_ms=0.0,
                consistency_model=ConsistencyModel.immediate,
                retrieved_count=1,
            ),
        )

    def format_context(self, result: RetrieveResult) -> str:
        return "\n".join(item["memory"] for item in result.raw)
