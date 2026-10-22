from __future__ import annotations

import pytest

from scripts.resolved_gt import setting_for


def test_setting_for_stable_answers_both_kinds_with_the_same_value() -> None:
    record = {"setting": "Detailed"}
    assert setting_for("pre_event", record) == "Detailed"
    assert setting_for("final", record) == "Detailed"


def test_setting_for_evolving_splits_pre_and_post() -> None:
    record = {"pre_setting": "Self-directed", "post_setting": "Reactive", "event_acc": 15}
    assert setting_for("pre_event", record) == "Self-directed"
    assert setting_for("final", record) == "Reactive"


@pytest.mark.parametrize(
    "record",
    [
        {"pre_setting": "Low", "event_acc": 30},
        {"setting": "Low", "pre_setting": "Low", "post_setting": "High", "event_acc": 30},
        {"setting": ""},
        {},
    ],
)
def test_setting_for_hard_errors_on_bad_shapes(record) -> None:
    with pytest.raises(ValueError):
        setting_for("final", record)


def test_setting_for_rejects_unknown_kind() -> None:
    with pytest.raises(ValueError, match="probe kind"):
        setting_for("accumulation", {"setting": "Low"})
