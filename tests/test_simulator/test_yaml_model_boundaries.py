import pytest

from simulator.schemas import PreferenceMatrix, SessionScript, YamlModelLoadError


ACTIVE_PREFERENCE_ATTRIBUTES = {
    "autonomy_level",
    "capability_boundary",
    "emotional_engagement",
    "guidance_level",
    "information_elicitation",
    "proactive_outreach",
    "process_visibility",
    "reasoning_visibility",
    "solution_breadth",
    "task_expansion",
    "tone_formality",
    "topic_management",
    "uncertainty_expression",
    "verbosity",
}


def test_runtime_preference_matrix_uses_current_contexts_and_attributes():
    matrix = PreferenceMatrix.from_yaml(
        "data/personas/user_a_preferences.yaml"
    )

    assert set(matrix.matrix) == {"work", "personal"}
    assert all(
        set(context_cells) == ACTIVE_PREFERENCE_ATTRIBUTES
        for context_cells in matrix.matrix.values()
    )
    assert "memory_privacy" not in ACTIVE_PREFERENCE_ATTRIBUTES


@pytest.mark.parametrize(
    ("content", "root_type", "value"),
    [
        ("", "NoneType", "None"),
        ("- one\n- two\n", "list", "['one', 'two']"),
    ],
)
def test_yaml_model_requires_mapping_root(
    tmp_path,
    content: str,
    root_type: str,
    value: str,
):
    path = tmp_path / "session.yaml"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(YamlModelLoadError) as exc_info:
        SessionScript.from_yaml(path)

    message = str(exc_info.value)
    assert str(path) in message
    assert f"root_type={root_type}" in message
    assert f"value={value}" in message


def test_yaml_model_validation_error_includes_path_and_field_value(tmp_path):
    path = tmp_path / "session.yaml"
    path.write_text(
        "session_id: acc_001\ncontext: unsupported_context\n",
        encoding="utf-8",
    )

    with pytest.raises(YamlModelLoadError) as exc_info:
        SessionScript.from_yaml(path)

    message = str(exc_info.value)
    assert str(path) in message
    assert "context" in message
    assert "unsupported_context" in message
