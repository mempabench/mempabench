"""The provider seam that recovers tool calls `deepseek-v4-flash` typed as prose.

regress5 acc_006: six of eleven PA turns came back with the model's own
`<｜DSML｜tool_calls>` syntax in `content` and an empty `tool_calls`, five of them
shipped the empty-reply template with no task call at all, and the session's
writes fell from 8 to 1. Every block below is the text that run really produced.
"""

from pa.providers.base import LLMResponse
from pa.providers.openai_compat_provider import (
    OpenAICompatProvider,
    _log_openrouter_call,
    convert_dsml_tool_calls,
)

#: regress5 acc_006 t16, verbatim. The whole turn was this and nothing else.
ACC_006_T16_READ = (
    '<｜DSML｜tool_calls>\n'
    '<｜DSML｜invoke name="mcp_state_document_read">\n'
    '<｜DSML｜parameter name="path" string="true">'
    'documents/research/citation_notes.md</｜DSML｜parameter>\n'
    '</｜DSML｜invoke>\n'
    '</｜DSML｜tool_calls>'
)

#: regress5 acc_006 t20, verbatim — the year fix that never ran. Two multi-line
#: `string="true"` values whose leading spaces and code fence are the anchor the
#: edit is matched on, so nothing about them may be normalised.
ACC_006_T20_UPDATE = (
    '<｜DSML｜tool_calls>\n'
    '<｜DSML｜invoke name="mcp_state_document_update">\n'
    '<｜DSML｜parameter name="path" string="true">'
    'documents/research/citation_notes.md</｜DSML｜parameter>\n'
    '<｜DSML｜parameter name="old_string" string="true">  year    = {2020},\n'
    '  eprint  = {2106.08829}\n'
    '}\n'
    '```\n'
    '\n'
    'Abelian influence-functional treatment. Borderline for Section 2: methods\n'
    'overlap (which-path leakage machinery), but no fusion space — cite as\n'
    'background contrast, not as support for the pathway argument.'
    '</｜DSML｜parameter>\n'
    '<｜DSML｜parameter name="new_string" string="true">  year    = {2021},\n'
    '  eprint  = {2106.08829}\n'
    '}\n'
    '```\n'
    '\n'
    'Abelian influence-functional treatment. Borderline for Section 2: methods\n'
    'overlap (which-path leakage machinery), but no fusion space — cite as\n'
    'background contrast, not as support for the pathway argument.'
    '</｜DSML｜parameter>\n'
    '</｜DSML｜invoke>\n'
    '</｜DSML｜tool_calls>'
)

_T20_OLD_STRING = (
    "  year    = {2020},\n"
    "  eprint  = {2106.08829}\n"
    "}\n"
    "```\n"
    "\n"
    "Abelian influence-functional treatment. Borderline for Section 2: methods\n"
    "overlap (which-path leakage machinery), but no fusion space — cite as\n"
    "background contrast, not as support for the pathway argument."
)

#: regress5 acc_006 t13, verbatim — and it stops mid-JSON, mid-parameter, with
#: no closing tag of any kind: the model ran out of completion tokens inside the
#: argument list. This is the block that must NOT become a call.
ACC_006_T13_TRUNCATED = (
    '<｜DSML｜tool_calls>\n'
    '<｜DSML｜invoke name="mcp_state_document_update">\n'
    '<｜DSML｜parameter name="path" string="true">'
    'documents/research/citation_notes.md</｜DSML｜parameter>\n'
    '<｜DSML｜parameter name="anchored_replacements" string="false">'
    '[{"anchor": "  pages   = {115312},", "old": "\\n  year    = {2020},", '
    '"new": "\\n  year    = {2021},"}, '
    '{"anchor": "check the fulltext before hanging the 2.2 sentence on it.", '
    '"old": "\\n\\n```bibtex\\n@article{okonkwo2024arxiv,\\n'
)

#: acc_006 t13's parameter with the block closed the way the model closes it
#: when it does not run out of tokens (t10, t20, t26 all close). `string="false"`
#: is the other half of the grammar: the value is JSON, not text.
ACC_006_T13_CLOSED = (
    '<｜DSML｜tool_calls>\n'
    '<｜DSML｜invoke name="mcp_state_document_update">\n'
    '<｜DSML｜parameter name="path" string="true">'
    'documents/research/citation_notes.md</｜DSML｜parameter>\n'
    '<｜DSML｜parameter name="anchored_replacements" string="false">'
    '[{"anchor": "  pages   = {115312},", "old": "\\n  year    = {2020},", '
    '"new": "\\n  year    = {2021},"}]</｜DSML｜parameter>\n'
    '</｜DSML｜invoke>\n'
    '</｜DSML｜tool_calls>'
)

#: regress4 acc_001 t32 — 7056 characters, all of it one closed `document_update`
#: invoke, none of it a reply. Its `replacement` value is cut here after the
#: first two lines; the three parameters and their spellings are as they arrived.
ACC_001_T32_ANCHORED = (
    '<｜DSML｜tool_calls>\n'
    '<｜DSML｜invoke name="mcp_state_document_update">\n'
    '<｜DSML｜parameter name="path" string="true">'
    'documents/research/citation_notes.md</｜DSML｜parameter>\n'
    '<｜DSML｜parameter name="anchor" string="true">'
    '## Formatted entries</｜DSML｜parameter>\n'
    '<｜DSML｜parameter name="replacement" string="true">## Formatted entries\n'
    '\n'
    '```bibtex</｜DSML｜parameter>\n'
    '</｜DSML｜invoke>\n'
    '</｜DSML｜tool_calls>'
)


def test_a_read_block_becomes_a_real_tool_call() -> None:
    """acc_006 t16: the whole reply was this, and the read never happened."""
    content, calls = convert_dsml_tool_calls(ACC_006_T16_READ)

    assert content is None
    assert [(call.id, call.name, call.arguments) for call in calls] == [
        ("dsml_1", "mcp_state_document_read",
         {"path": "documents/research/citation_notes.md"}),
    ]


def test_the_prose_around_a_block_survives_it() -> None:
    """acc_006 t10's shape: a sentence to the user, then the call typed out."""
    prose = "Good, I have all the decisions. Let me proceed with the four edits now."

    content, calls = convert_dsml_tool_calls(f"{prose}\n\n{ACC_006_T16_READ}")

    assert content == prose
    assert "DSML" not in content
    assert len(calls) == 1


def test_multi_line_string_parameters_arrive_verbatim() -> None:
    """acc_006 t20 — the anchored edit whose whitespace is what it matches on."""
    _content, calls = convert_dsml_tool_calls(ACC_006_T20_UPDATE)

    assert len(calls) == 1
    assert calls[0].name == "mcp_state_document_update"
    assert calls[0].arguments["path"] == "documents/research/citation_notes.md"
    assert calls[0].arguments["old_string"] == _T20_OLD_STRING
    assert calls[0].arguments["new_string"] == _T20_OLD_STRING.replace("2020", "2021")


def test_a_json_parameter_is_decoded_not_kept_as_text() -> None:
    """`string="false"` is the model saying the value is JSON (acc_006 t13)."""
    _content, calls = convert_dsml_tool_calls(ACC_006_T13_CLOSED)

    assert calls[0].arguments["anchored_replacements"] == [
        {
            "anchor": "  pages   = {115312},",
            "old": "\n  year    = {2020},",
            "new": "\n  year    = {2021},",
        },
    ]


def test_the_acc_001_block_becomes_a_call() -> None:
    """acc_001 t32: three parameters, one closed invoke, one recovered write."""
    content, calls = convert_dsml_tool_calls(ACC_001_T32_ANCHORED)

    assert content is None
    assert calls[0].name == "mcp_state_document_update"
    assert set(calls[0].arguments) == {"path", "anchor", "replacement"}
    assert calls[0].arguments["anchor"] == "## Formatted entries"


def test_a_truncated_block_is_left_for_the_renderer_to_strip() -> None:
    """acc_006 t13 — half an argument list run against the user's documents is
    worse than a call nobody makes. The block stays in `content` untouched, so
    the IX filter drops it and `ix_markup_stripped` records what was lost."""
    prose = "Applying the two year fixes."

    content, calls = convert_dsml_tool_calls(f"{prose}\n\n{ACC_006_T13_TRUNCATED}")

    assert calls == []
    assert content == f"{prose}\n\n{ACC_006_T13_TRUNCATED}"


def test_a_malformed_invoke_disqualifies_the_whole_block() -> None:
    """One good invoke beside one the grammar cannot close: neither runs."""
    block = (
        '<｜DSML｜tool_calls>\n'
        '<｜DSML｜invoke name="mcp_state_document_read">\n'
        '<｜DSML｜parameter name="path" string="true">a.md</｜DSML｜parameter>\n'
        '</｜DSML｜invoke>\n'
        '<｜DSML｜invoke name="mcp_state_document_update">\n'
        '<｜DSML｜parameter name="path" string="true">b.md\n'
        '</｜DSML｜tool_calls>'
    )

    content, calls = convert_dsml_tool_calls(block)

    assert calls == []
    assert content == block


def test_ids_are_minted_in_order_and_do_not_move() -> None:
    two_invokes = (
        '<｜DSML｜tool_calls>\n'
        '<｜DSML｜invoke name="mcp_state_document_read">\n'
        '<｜DSML｜parameter name="path" string="true">a.md</｜DSML｜parameter>\n'
        '</｜DSML｜invoke>\n'
        '<｜DSML｜invoke name="mcp_state_document_read">\n'
        '<｜DSML｜parameter name="path" string="true">b.md</｜DSML｜parameter>\n'
        '</｜DSML｜invoke>\n'
        '</｜DSML｜tool_calls>'
    )

    _content, calls = convert_dsml_tool_calls(two_invokes)

    assert [call.id for call in calls] == ["dsml_1", "dsml_2"]
    assert [call.arguments["path"] for call in calls] == ["a.md", "b.md"]
    assert convert_dsml_tool_calls(two_invokes)[1][0].id == "dsml_1"


def test_the_ascii_pipe_spelling_is_matched_too() -> None:
    """The renderer's stripper tolerates both bars; the parser has to agree, or
    a spelling it strips is a spelling it also silently swallows."""
    ascii_block = (
        '<|DSML|>tool_calls>\n'
        '<|DSML|>invoke name="mcp_state_document_read">\n'
        '<|DSML|>parameter name="path" string="true">a.md</|DSML|>parameter>\n'
        '</|DSML|>invoke>\n'
        '</|DSML|>tool_calls>'
    )

    _content, calls = convert_dsml_tool_calls(ascii_block)

    assert [call.name for call in calls] == ["mcp_state_document_read"]


def test_text_with_no_markup_is_returned_unchanged() -> None:
    answer = "The Brandao flag is unresolved: their pointer basis is not our P1 label."

    assert convert_dsml_tool_calls(answer) == (answer, [])
    assert convert_dsml_tool_calls(None) == (None, [])


# ---------------------------------------------------------------------------
# The seam itself
# ---------------------------------------------------------------------------


def _parsed(content: str) -> LLMResponse:
    provider = object.__new__(OpenAICompatProvider)
    return provider._recover_dsml_tool_calls(provider._parse({
        "id": "request_1",
        "model": "deepseek/deepseek-v4-flash",
        "choices": [{
            "finish_reason": "stop",
            "message": {"role": "assistant", "content": content, "tool_calls": []},
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20},
    }))


def test_the_seam_moves_the_call_off_the_content_plane() -> None:
    response = _parsed(f"Reading the file.\n\n{ACC_006_T16_READ}")

    assert response.content == "Reading the file."
    assert [call.name for call in response.tool_calls] == ["mcp_state_document_read"]
    assert response.has_tool_calls
    assert response.dsml_converted == 1
    # Left alone on purpose: the runner branches on `has_tool_calls`, and the
    # audit line should keep saying what the provider actually answered.
    assert response.finish_reason == "stop"


def test_the_seam_leaves_a_well_behaved_response_alone() -> None:
    response = _parsed("Both years are already correct.")

    assert response.content == "Both years are already correct."
    assert response.tool_calls == []
    assert response.dsml_converted == 0


def test_the_call_line_counts_the_recovered_calls(capsys) -> None:
    _log_openrouter_call(
        "deepseek/deepseek-v4-flash", 0.0,
        LLMResponse(content=None, usage={"prompt_tokens": 1, "completion_tokens": 2},
                    dsml_converted=2),
        cost=0.0004,
    )

    line = capsys.readouterr().err.strip()
    assert "dsml_tool_calls=2" in line
    assert line.startswith("openrouter_call model=deepseek/deepseek-v4-flash")
    assert line.endswith(" finish_reason=stop")


def test_the_call_line_says_why_the_model_stopped(capsys) -> None:
    _log_openrouter_call(
        "google/gemini-3.8-flash", 0.0,
        LLMResponse(content=None, usage={"prompt_tokens": 1, "completion_tokens": 4092},
                    finish_reason="length"),
        cost=None,
    )

    assert capsys.readouterr().err.strip().endswith(" finish_reason=length")


def test_the_call_line_says_nothing_when_nothing_was_recovered(capsys) -> None:
    _log_openrouter_call(
        "deepseek/deepseek-v4-flash", 0.0,
        LLMResponse(content="ok", usage={"prompt_tokens": 1, "completion_tokens": 2}),
        cost=None,
    )

    assert "dsml" not in capsys.readouterr().err
