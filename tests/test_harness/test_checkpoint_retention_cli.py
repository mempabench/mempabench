from types import SimpleNamespace

import pytest

from scripts.run_orchestrator import _build_arg_parser, _retention_mainline_ids


def test_retention_scope_includes_all_real_mainline_steps_only():
    steps = []
    for number in range(1, 11):
        steps.append(SimpleNamespace(
            step_id=f"step_{number}",
            kind="forced_event" if number == 5 else "accumulation",
            placeholder=False,
        ))
        steps.append(SimpleNamespace(step_id=f"gap_{number}", kind="accumulation", placeholder=True))
        steps.append(SimpleNamespace(step_id=f"pre_{number}", kind="pre_event_probe", placeholder=False))
    plan = SimpleNamespace(steps=steps)
    assert _retention_mainline_ids(plan) == frozenset(f"step_{n}" for n in range(1, 11))


@pytest.mark.parametrize("value", ["0", "-1", "bad"])
def test_retention_cli_rejects_invalid_intervals(value):
    with pytest.raises(SystemExit):
        _build_arg_parser().parse_args(["--retain-checkpoints-every", value])


def test_retention_cli_is_opt_in():
    parser = _build_arg_parser()
    assert parser.parse_args([]).retain_checkpoints_every is None
    assert parser.parse_args(["--retain-checkpoints-every", "5"]).retain_checkpoints_every == 5
