"""Probe budgets survive restarts and never erase an unresolved paid intent."""
import json

import pytest

from harness.probe_attempts import ProbeAttempts


def test_three_total_attempts_survive_process_reconstruction(tmp_path):
    for number in (1, 2, 3):
        attempt = ProbeAttempts(tmp_path, "fin_01")
        assert attempt.next_number() == number
        assert attempt.begin() == number
        assert json.loads(attempt.path.read_text())["attempts"][-1]["status"] == "started"
        attempt.finish(error=RuntimeError("whole-IX-void"))
    with pytest.raises(RuntimeError, match="three counted"):
        ProbeAttempts(tmp_path, "fin_01").begin()
    assert len(json.loads(attempt.path.read_text())["attempts"]) == 3


def test_unknown_attempt_blocks_replay_but_does_not_block_another_question(tmp_path):
    first = ProbeAttempts(tmp_path, "fin_01")
    first.begin()
    before = first.path.read_bytes()
    with pytest.raises(RuntimeError, match="unknown outcome"):
        ProbeAttempts(tmp_path, "fin_01").begin()
    assert first.path.read_bytes() == before
    assert ProbeAttempts(tmp_path, "fin_02").begin() == 1


def test_success_survives_restart_and_cannot_be_regenerated(tmp_path):
    first = ProbeAttempts(tmp_path, "pre_01")
    first.begin()
    first.finish()
    with pytest.raises(RuntimeError, match="successful"):
        ProbeAttempts(tmp_path, "pre_01").begin()


def test_finished_success_cannot_be_changed_to_failure(tmp_path):
    first = ProbeAttempts(tmp_path, "fin_01")
    first.begin()
    first.finish()
    before = first.path.read_bytes()
    with pytest.raises(RuntimeError):
        first.finish(error=RuntimeError("late unrelated error"))
    assert first.path.read_bytes() == before


def test_stale_instance_cannot_overwrite_another_started_intent(tmp_path):
    first = ProbeAttempts(tmp_path, "fin_01")
    stale = ProbeAttempts(tmp_path, "fin_01")
    first.begin()
    before = first.path.read_bytes()
    with pytest.raises(RuntimeError, match="unknown outcome"):
        stale.begin()
    assert first.path.read_bytes() == before


def test_attempts_are_independent_between_repeats_and_questions(tmp_path):
    for repeat in ("repeat02", "repeat03"):
        for question in ("pre_01", "fin_01"):
            attempt = ProbeAttempts(tmp_path / repeat, question)
            assert attempt.begin() == 1
            attempt.finish()
    assert len(list(tmp_path.glob("*/probe_attempts/*.json"))) == 4
