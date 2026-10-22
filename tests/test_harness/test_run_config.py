"""configs/run_config.yaml: every key's default equals the shipped file."""

from __future__ import annotations

import pytest
import yaml

from harness import run_config
from harness.run_config import (
    FailurePolicy,
    RunConfig,
    effective_failure_policy,
    load_run_config,
)


def test_the_shipped_file_equals_the_code_defaults():
    assert load_run_config(run_config.DEFAULT_RUN_CONFIG_PATH) == RunConfig()


def test_defaults_skip_only_ix_exhaustion_and_keep_todays_budgets():
    config = RunConfig()
    assert config.failure_policy == FailurePolicy(
        skip_steps={},
        auto_skip_on_ix_exhaustion=True,
        max_scene_attempts=1,
        partial_skip_pending_memory="drop",
    )
    assert (config.probe.attempt_limit, config.probe.total_attempt_limit) == (3, 6)
    assert config.probe.network_failures_count is False
    assert config.repeat.rounds == [2, 3]
    assert config.budgets.pa_max_tokens == 24576
    assert config.budgets.mem0_extraction_max_output_tokens == 16000
    assert config.providers.probe_phase is None


def test_a_missing_key_keeps_its_default_and_an_unknown_key_is_refused(tmp_path):
    path = tmp_path / "run_config.yaml"
    path.write_text(yaml.safe_dump({"budgets": {"pa_max_tokens": 4096}}))
    config = load_run_config(path)
    assert config.budgets.pa_max_tokens == 4096
    assert config.budgets.memory_retrieval_max_items == 10
    path.write_text(yaml.safe_dump({"budget": {}}))
    with pytest.raises(ValueError):
        load_run_config(path)


def test_the_environment_names_another_file(tmp_path, monkeypatch):
    path = tmp_path / "smoke.yaml"
    path.write_text(yaml.safe_dump({"probe": {"attempt_limit": 2}}))
    monkeypatch.setenv(run_config.RUN_CONFIG_ENV, str(path))
    assert load_run_config().probe.attempt_limit == 2
    monkeypatch.setenv(run_config.RUN_CONFIG_ENV, str(tmp_path / "missing.yaml"))
    with pytest.raises(FileNotFoundError):
        load_run_config()


def test_a_run_directory_policy_replaces_the_global_one(tmp_path):
    config = RunConfig()
    assert effective_failure_policy(tmp_path, config) == config.failure_policy
    (tmp_path / run_config.FAILURE_POLICY_OVERRIDE_NAME).write_text(
        yaml.safe_dump({"skip_steps": {"acc_007": "retain_partial"}})
    )
    policy = effective_failure_policy(tmp_path, config)
    assert policy.skip_steps == {"acc_007": "retain_partial"}
    assert policy.max_scene_attempts == 1
