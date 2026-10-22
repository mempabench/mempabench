"""Feature rubric files (`judge/prompts/features/<attr>.md`) carry only live settings.

Retired middle settings (feedback_no_ambiguous_middle_settings) and struck-through
prose must not reach the judge prompt, and neither does the human commentary each
file keeps below `COMMENTARY_HEADING` (§4.5 block ⑥). Run:
    python -m pytest tests/test_judge/test_feature_files.py -q
"""

from __future__ import annotations

import re

import pytest

from data.personas.coverage import TAXONOMY
from judge.evaluator import COMMENTARY_HEADING, FEATURES_DIR, rubric_markdown

RETIRED_SETTINGS = ("Moderate", "Balanced", "Calibrated", "Medium")
#: Strings that only ever occur in the commentary half of a feature file.
COMMENTARY_MARKERS = ("Pattern_to_Preference_Rubrics.md", "阈值的地位", "与现有 rubric 的关系")


def _feature_text(attribute: str) -> str:
    return (FEATURES_DIR / f"{attribute}.md").read_text(encoding="utf-8")


def test_every_taxonomy_attribute_has_a_feature_file():
    assert sorted(p.stem for p in FEATURES_DIR.glob("*.md")) == sorted(TAXONOMY)


@pytest.mark.parametrize("attribute", sorted(TAXONOMY))
def test_no_retired_setting_or_strikethrough(attribute):
    text = _feature_text(attribute)
    assert "~~" not in text
    for retired in RETIRED_SETTINGS:
        assert not re.search(rf"\b{retired}\b", text), f"{attribute}.md names retired setting {retired}"


@pytest.mark.parametrize("attribute", sorted(TAXONOMY))
def test_every_live_setting_is_mentioned(attribute):
    text = _feature_text(attribute)
    for setting in TAXONOMY[attribute]:
        assert f"`{setting}`" in text, f"{attribute}.md never mentions `{setting}`"


# ----------------------------------------------------- judge-facing / commentary


@pytest.mark.parametrize("attribute", sorted(TAXONOMY))
def test_commentary_heading_appears_exactly_once(attribute):
    assert _feature_text(attribute).count(COMMENTARY_HEADING) == 1


@pytest.mark.parametrize("attribute", sorted(TAXONOMY))
def test_block_six_keeps_the_table_and_drops_the_commentary(attribute):
    block = rubric_markdown(attribute)
    assert "| Setting | 可观测特征 | evidence 类型 |" in block
    assert COMMENTARY_HEADING not in block
    for marker in COMMENTARY_MARKERS:
        assert marker not in block, f"{attribute}: block ⑥ still carries commentary marker {marker}"


@pytest.mark.parametrize("attribute", sorted(TAXONOMY))
def test_block_six_still_names_every_setting(attribute):
    # Past the generated `**属性** … 现行 setting 列表` line, so the feature text itself
    # has to name each setting.
    body = rubric_markdown(attribute).split("\n\n", 1)[1]
    for setting in TAXONOMY[attribute]:
        assert f"`{setting}`" in body, f"{attribute}: judge-facing half never names `{setting}`"
