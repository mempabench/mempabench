from simulator.skills import load_actor_skill_spec


def _normalized(attribute: str, setting: str) -> str:
    return " ".join(load_actor_skill_spec(attribute, setting).split()).casefold()


def test_process_visibility_full_narration_covers_every_execution_step() -> None:
    # The narration reaches the user as one line per step, stacked at the top of a
    # single finished message. The simulated user must not learn that those steps
    # were tool calls, so the spec says "step" and the contract follows it.
    body = _normalized("process_visibility", "Full narration")
    assert "one short line for every step it took" in body
    assert "no step passes unreported" in body


def test_process_visibility_silent_is_result_only() -> None:
    # There are no mid-task messages to suppress: one message per turn, and Silent
    # drops the opening line and the closing one from it. The old wording promised
    # the user updates "in between" that never exist.
    body = _normalized("process_visibility", "Silent")
    assert "leaves out is the line about starting and the line about being done" in body
    assert "still delivers the finished result" in body


def test_autonomy_scope_not_reporting_separates_high_settings() -> None:
    self_directed = _normalized("autonomy_level", "Self-directed")
    autonomous = _normalized("autonomy_level", "Autonomous")
    assert "clear, routine, in-scope actions" in self_directed
    assert "the pa still reports the finished result" in self_directed
    assert "delegated in-scope decisions" in autonomous
    assert "including non-routine ones" in autonomous
    assert "the scope of what the pa may decide is wider" in autonomous


def test_autonomy_specs_keep_proposal_separate_from_execution() -> None:
    reactive = _normalized("autonomy_level", "Reactive")
    self_directed = _normalized("autonomy_level", "Self-directed")

    assert "does not need permission to inspect" in reactive
    assert "before carrying out each substantive action" in reactive
    assert "gather information and do preparatory work" in self_directed
    assert "key decision and at anything with external impact" in self_directed


def test_topic_management_settings_complete_every_user_raised_topic() -> None:
    for setting in ("Organize", "One-at-a-time"):
        body = _normalized("topic_management", setting)
        assert "every" in body
        assert "topic" in body
        assert "does not drop anything" in body or "never permits abandoning" in body or (
            "not whether work can be dropped" in body
        )


def test_capability_handoff_does_not_require_an_external_recipient() -> None:
    handoff = _normalized("capability_boundary", "Find and hand off")
    alternatives = _normalized("capability_boundary", "Suggest alternatives")
    assert "does not mean finding another person" in handoff
    assert "blocked decision or action back in your hands" in handoff
    assert "agent-side recovery" in alternatives


def test_low_solution_breadth_keeps_task_competence_fixed() -> None:
    body = _normalized("solution_breadth", "Low")
    assert "accuracy and competence do not change" in body
    assert "single best answer" in body


def test_task_expansion_and_outreach_are_separated_by_active_request() -> None:
    body = _normalized("task_expansion", "High")
    assert "while a request is active" in body
    assert "while no request is active" in body
    assert "may still belong to the same longer task" in body


def test_low_proactive_outreach_waits_for_user() -> None:
    body = _normalized("proactive_outreach", "Low")
    assert "wait until you ask or come back to it" in body
    assert "unless silence" not in body


def test_every_spec_declares_base_def_and_asymmetry() -> None:
    """The two contracts added by the de-AI pass: base-rate licence + PA-side framing."""
    import pathlib

    root = pathlib.Path("data/actor_skill_specs")
    specs = sorted(root.glob("*/*.md"))
    assert len(specs) == 32
    for spec in specs:
        text = spec.read_text()
        assert "## Base Def" in text, spec
        assert "**Asymmetry**" in text, spec
        assert "## Boundaries" not in text, spec
