"""The setting-definition table, `configs/setting_definitions.yaml`, typed.

One row per attribute x setting, in document order (the order of
`docs/definitions/PA_Interaction_Preference.md`, of `data/personas/coverage.py`
`TAXONOMY`, and of the judge numbering in
`docs/definitions/MemPABench_Attribute_Feature_Tables.md`). `specs.py` builds
`IX_PASSIVE_SPECS` from it at import time, so `pa_instruction` is the text the PA
is handed and enters `spec_sha256`; `user_sees` / `judge_observable` /
`judge_evidence` are the table cells `scripts/render_setting_definitions.py`
writes into the two definition documents. Every cell is one line of markdown
table content: no line break, no `|`.

Standard library, pydantic and yaml only. `render_types.py` has to stay
importable without this module and `specs.py` imports it, so nothing here
imports from `pa.agent.ix`.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

DEFAULT_SETTING_DEFINITIONS_PATH = (
    Path(__file__).resolve().parents[3] / "configs/setting_definitions.yaml"
)

ContractClass = Literal[
    "output_realization",
    "interaction_sequence",
    "action_scope",
    "triggered_interaction",
]


class SettingDefinitionsError(ValueError):
    """Raised when the table is missing or does not have the declared shape."""


def _table_cell(value: str) -> str:
    if not value.strip():
        raise ValueError("cell is empty")
    if "\n" in value or "\r" in value:
        raise ValueError("cell contains a line break")
    if "|" in value:
        raise ValueError("cell contains '|', which would split the markdown table")
    if value != value.strip():
        raise ValueError("cell has leading or trailing whitespace")
    return value


class SettingRow(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    name: str
    pa_instruction: str
    user_sees: str
    judge_observable: str
    judge_evidence: str
    #: What the wrapper releases or holds under this setting (plan §3.1). Filled in U6.
    wrapper: str | None = None

    @field_validator(
        "name", "pa_instruction", "user_sees", "judge_observable", "judge_evidence", "wrapper"
    )
    @classmethod
    def _one_line(cls, value: str | None) -> str | None:
        return None if value is None else _table_cell(value)


class AttributeRow(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    id: str
    label: str
    contract_class: ContractClass
    decision: str
    #: Simple counts that recur in several texts (plan §2), keyed by a name of
    #: the author's choosing; empty until a number really appears in every text.
    thresholds: dict[str, int] = Field(default_factory=dict)
    settings: list[SettingRow]

    @field_validator("id")
    @classmethod
    def _snake_case(cls, value: str) -> str:
        if not value or not all(ch.islower() or ch == "_" for ch in value):
            raise ValueError(f"attribute id must be lower_snake_case: {value!r}")
        return value

    @field_validator("label", "decision")
    @classmethod
    def _one_line(cls, value: str) -> str:
        return _table_cell(value)

    @field_validator("thresholds")
    @classmethod
    def _positive(cls, value: dict[str, int]) -> dict[str, int]:
        for key, number in value.items():
            if number <= 0:
                raise ValueError(f"threshold {key!r} must be a positive count, got {number}")
        return value

    @model_validator(mode="after")
    def _settings_shape(self) -> AttributeRow:
        if len(self.settings) < 2:
            raise ValueError(f"{self.id}: an attribute needs at least two settings")
        names = [setting.name for setting in self.settings]
        if len(set(names)) != len(names):
            raise ValueError(f"{self.id}: duplicate setting name in {names}")
        return self


class SettingDefinitions(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    schema_version: Literal[1]
    source_note: str
    attributes: list[AttributeRow]

    @model_validator(mode="after")
    def _unique_ids(self) -> SettingDefinitions:
        ids = [attribute.id for attribute in self.attributes]
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate attribute id in {ids}")
        return self


@lru_cache(maxsize=8)
def load_setting_definitions(
    path: Path = DEFAULT_SETTING_DEFINITIONS_PATH,
) -> SettingDefinitions:
    """Load and validate the table; cached per path for the life of the process."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SettingDefinitionsError(
            f"setting definitions are unavailable: {path} ({type(exc).__name__}: {exc})"
        ) from exc
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise SettingDefinitionsError(f"{path}: invalid YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise SettingDefinitionsError(f"{path}: the document root must be a mapping")
    try:
        return SettingDefinitions.model_validate(data)
    except ValidationError as exc:
        raise SettingDefinitionsError(f"{path}: {exc}") from exc
