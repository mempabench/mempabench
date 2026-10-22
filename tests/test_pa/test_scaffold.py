"""Structural tests for PA scaffold — no API keys needed."""

import asyncio
import tempfile
from pathlib import Path

from pa.agent.loop import AgentLoop
from pa.agent.context import ContextBuilder
from pa.agent.memory import MarkdownMemoryStore, MemoryConsolidator
from pa.agent.runner import AgentRunner
from pa.agent.tools.base import Tool
from pa.agent.ix import (
    IX_PASSIVE_SPECS,
    IXPreferenceTool,
    TurnToolRegistry,
    ix_tool_names_for,
)
from pa.agent.ix.collector import (
    CERTAINTY_CLAIM_SOURCE_REMINDER,
    turn_reminder_message,
)
from pa.agent.ix.events import IXSelection
from pa.agent.ix.rationale import CERTAINTY_BLOCK_INSTRUCTION
from pa.agent.ix.tools import IX_RENDER_TOOL_SPECS
from pa.agent.tools.registry import ToolRegistry
from pa.bus.events import InboundMessage, OutboundMessage
from pa.bus.queue import MessageBus
from pa.command.router import CommandRouter
from pa.session.manager import Session, SessionManager
from pa.providers.base import LLMProvider, LLMResponse


class DummyTool(Tool):
    name = "dummy"
    description = "Dummy test tool"
    parameters = {"type": "object", "properties": {}, "required": []}

    async def execute(self, **kwargs):
        return "ok"


def test_all_imports():
    """All PA modules import without error."""
    assert AgentLoop is not None
    assert ContextBuilder is not None
    assert MarkdownMemoryStore is not None
    assert AgentRunner is not None
    assert ToolRegistry is not None
    assert CommandRouter is not None
    assert SessionManager is not None
    print("OK: All PA modules import successfully")


def test_context_benchmark_identity():
    """Context builder supports benchmark identity override."""
    with tempfile.TemporaryDirectory() as td:
        ws = Path(td)
        ctx = ContextBuilder(ws)

        # Default identity
        default_prompt = ctx.build_system_prompt()
        assert "nanobot" in default_prompt.lower()

        # Override with benchmark identity
        ctx.set_benchmark_identity("You are a personal assistant for benchmark testing.")
        custom_prompt = ctx.build_system_prompt()
        assert "benchmark testing" in custom_prompt
        assert "nanobot" not in custom_prompt.split("---")[0].lower()

    print("OK: Benchmark identity override works")


def test_context_uses_prompt_workspace_alias(monkeypatch, tmp_path: Path):
    """Sensitive run directory names can be hidden from provider-visible prompts."""
    workspace = tmp_path / "sensitive_provider_run" / "pa_workspace" / "user_a"
    workspace.mkdir(parents=True)
    monkeypatch.setenv("NANOBOT_PROMPT_WORKSPACE", "/workspace/user_a")

    prompt = ContextBuilder(workspace).build_system_prompt()

    # The memory files are named by basename only; a printed resolvable path is
    # what the model copies into document_read (regress1/regress2 acc_004).
    assert "MEMORY.md" not in prompt
    assert "/memory/" not in prompt
    assert "sensitive_provider_run" not in prompt
    assert str(workspace) not in prompt


def test_context_auto_aliases_benchmark_run_workspace(monkeypatch, tmp_path: Path):
    """Benchmark run paths are hidden from provider-visible prompts by default."""
    monkeypatch.delenv("NANOBOT_PROMPT_WORKSPACE", raising=False)
    sensitive_run_id = ("ten" + "cent") + "_memory_token_surface_20260723_003"
    workspace = (
        tmp_path
        / "workspace"
        / "runs"
        / "azure_openai_gpt-5.5"
        / "nanobot_file_memory"
        / "user_a"
        / sensitive_run_id
        / "pa_workspace"
        / "user_a"
    )
    workspace.mkdir(parents=True)

    prompt = ContextBuilder(workspace).build_system_prompt()

    assert "Your workspace is at: /workspace/user_a" in prompt
    assert "/workspace/user_a/memory/MEMORY.md" not in prompt
    assert "ten" + "cent" not in prompt.lower()
    assert str(workspace) not in prompt


def test_context_auto_aliases_diagnostic_workspace(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("NANOBOT_PROMPT_WORKSPACE", raising=False)
    workspace = (
        tmp_path
        / "workspace"
        / "diagnostics"
        / "memory_online_smoke"
        / "openai_gpt-5.5"
        / "nanobot_file_memory"
        / "user_a"
        / "diagnostic_001"
        / "pa_workspace"
        / "user_a"
    )
    workspace.mkdir(parents=True)

    prompt = ContextBuilder(workspace).build_system_prompt()

    assert "Your workspace is at: /workspace/user_a" in prompt
    assert str(workspace) not in prompt


def test_context_keeps_ordinary_workspace_path(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("NANOBOT_PROMPT_WORKSPACE", raising=False)
    workspace = tmp_path / "ordinary" / "user_a"
    workspace.mkdir(parents=True)

    prompt = ContextBuilder(workspace).build_system_prompt()

    assert f"Your workspace is at: {workspace.resolve()}" in prompt


def test_prompt_never_sends_the_pa_to_a_tool_for_its_own_memory(
    monkeypatch, tmp_path: Path
):
    """In benchmark mode the PA holds only `mcp_state_*` and `IX_*` tools.

    The prompt used to say MEMORY.md was where to "write important facts" and to
    open a SKILL.md with the `read_file` tool. Neither tool exists here, so
    across the pre_run_test_* sessions the PA spent 21 calls in 10 sessions
    pushing its own memory paths through `document_read` / `document_update`.
    MEMORY.md is injected verbatim under "# Memory" and written only by the
    consolidation step, so the prompt has to say so.
    """
    monkeypatch.delenv("NANOBOT_PROMPT_WORKSPACE", raising=False)
    workspace = tmp_path / "ordinary" / "user_a"
    (workspace / "memory").mkdir(parents=True)
    (workspace / "memory" / "MEMORY.md").write_text(
        "- Prefers short replies\n", encoding="utf-8"
    )
    skill = workspace / "skills" / "demo"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        "---\nname: demo\ndescription: A demo skill.\n---\n\nBody.\n", encoding="utf-8"
    )

    prompt = ContextBuilder(workspace).build_system_prompt()

    # The skills block is rendered, so the read_file check below has teeth.
    assert "<skills>" in prompt
    assert "read_file" not in prompt
    assert "write important facts here" not in prompt

    # The memory files are described as maintained for the PA, not as targets.
    assert "maintained for you automatically" in prompt
    assert '"# Memory" section' in prompt
    assert "/memory/" not in prompt and "memory/MEMORY.md" not in prompt
    # The ban names the basenames, not the two absolute paths: live `acc_039`
    # walked straight past a path-shaped ban with the relative
    # `memory/MEMORY.md` and reached `document_read`.
    assert "no memory file for you to read, search or write" in prompt
    assert "MEMORY.md" not in prompt and "HISTORY.md" not in prompt
    assert "under any name or path" in prompt

    # ... and their content is already in the prompt, so nothing needs fetching.
    assert "- Prefers short replies" in prompt


def test_prompt_says_an_anchored_replace_is_the_only_way_to_edit_a_document(
    monkeypatch, tmp_path: Path
):
    """Round-2 `acc_003` t=6: a 9.1 KB tool-call blob shipped as prose.

    The model wrote out a whole rewritten file as an argument to
    `mcp_state_document_write`, a tool that does not exist, emitted it as text
    instead of calling anything, and the turn executed nothing. The tool list
    it holds has no whole-file writer at all, so the prompt says so.
    """
    monkeypatch.delenv("NANOBOT_PROMPT_WORKSPACE", raising=False)
    workspace = tmp_path / "ordinary" / "user_a"
    workspace.mkdir(parents=True)

    prompt = ContextBuilder(workspace).build_system_prompt()

    assert (
        "`document_update` (anchored replace) is the only way to change an "
        "existing document; there is no whole-file write tool"
    ) in prompt


def test_session_checkpoint():
    """Session checkpoint and restore work correctly."""
    with tempfile.TemporaryDirectory() as td:
        ws = Path(td)
        sm = SessionManager(ws)

        s = sm.create("bench:001")
        s.add_message("user", "Hello")
        s.add_message("assistant", "Hi!")
        s.add_message("user", "I like brief answers")
        sm.save(s)

        # Checkpoint at 3 messages
        cp_dir = Path(td) / "checkpoints" / "s1"
        sm.checkpoint(s, cp_dir)

        # Add more
        s.add_message("assistant", "Noted.")
        sm.save(s)
        assert len(s.messages) == 4

        # Restore — should have 3
        restored = sm.restore_checkpoint("bench:001_eval", cp_dir)
        assert restored is not None
        assert len(restored.messages) == 3

    print("OK: Session checkpoint/restore works")


def test_tool_registry():
    """Tools register and list correctly."""
    reg = ToolRegistry()
    reg.register(DummyTool())

    assert reg.get("dummy") is not None
    defs = reg.get_definitions()
    assert len(defs) == 1

    print("OK: Tool registry works")


def test_turn_tool_registry_exposes_active_ix_only():
    """A turn gets IX tools only for active control-unaffected preferences."""
    reg = ToolRegistry()
    reg.register(DummyTool())

    turn = TurnToolRegistry(
        reg,
        [
            "guidance_level",
            "topic_management",
            "uncertainty_expression",
            "process_visibility",
            "autonomy_level",
            "solution_breadth",
            "proactive_outreach",
            "task_expansion",
            "capability_boundary",
            "information_elicitation",
        ],
    )
    names = turn.tool_names

    assert "dummy" in names
    assert "IX_guidance_level" in names
    assert "IX_uncertainty_expression" in names
    assert "IX_process_visibility" in names
    assert "IX_topic_management" in names
    assert "IX_autonomy_level" in names
    assert "IX_solution_breadth" in names
    assert "IX_proactive_outreach" in names
    assert "IX_task_expansion" in names
    assert "IX_capability_boundary" in names
    assert "IX_information_elicitation" in names
    assert ix_tool_names_for(["reasoning_visibility"]) == ["IX_reasoning_visibility"]
    assert ix_tool_names_for(["process_visibility"]) == ["IX_process_visibility"]
    assert ix_tool_names_for(["topic_management"]) == ["IX_topic_management"]

    defs = turn.get_definitions()
    tool_names = [d["function"]["name"] for d in defs]
    assert tool_names == [
        "dummy",
        "IX_guidance_level",
        "IX_topic_management",
        "IX_uncertainty_expression",
        "IX_process_visibility",
        "IX_autonomy_level",
        "IX_solution_breadth",
        "IX_proactive_outreach",
        "IX_task_expansion",
        "IX_capability_boundary",
        "IX_information_elicitation",
    ]

    print("OK: Turn tool registry exposes active IX tools correctly")


def test_ix_tool_descriptions_define_settings():
    """IX tools expose setting semantics, not just enum labels."""
    for spec in IX_PASSIVE_SPECS.values():
        assert set(spec.setting_descriptions) == set(spec.settings)
        description = IXPreferenceTool(spec).description
        for setting in spec.settings:
            assert f"{setting}:" in description
            assert spec.setting_descriptions[setting] in description

    breadth_description = IXPreferenceTool(IX_PASSIVE_SPECS["solution_breadth"]).description
    assert "Low: Present exactly one option" in breadth_description
    assert "Medium" not in breadth_description
    assert "High: Present at least two genuinely viable options" in breadth_description

    outreach_description = IXPreferenceTool(IX_PASSIVE_SPECS["proactive_outreach"]).description
    assert "Low: No request is active and the renderer suppresses the contact" in outreach_description
    assert "High: No request is active and the renderer delivers your message" in outreach_description
    proactive_description = IXPreferenceTool(
        IX_PASSIVE_SPECS["proactive_outreach"], proactive_event=True
    ).description
    assert "Low requires suppress" in proactive_description
    assert "High requires send" in proactive_description

    expansion_description = IXPreferenceTool(IX_PASSIVE_SPECS["task_expansion"]).description
    assert "Low: Deliver the thing that was asked for and carry out no adjacent step" in expansion_description
    assert "Medium" not in expansion_description
    assert "High: Deliver the thing that was asked for and carry out at least one adjacent" in expansion_description
    assert "outside the items named in the request" in expansion_description


def test_proactive_ix_tool_rejects_setting_action_contradictions():
    async def run() -> None:
        tool = IXPreferenceTool(
            IX_PASSIVE_SPECS["proactive_outreach"], proactive_event=True
        )

        low_send = await tool.execute(
            setting="Low",
            outreach_action="send",
            outreach_message="A deadline changed.",
        )
        high_suppress = await tool.execute(
            setting="High",
            outreach_action="suppress",
            outreach_message="",
        )
        # The taxonomy is two settings since 2026-09-05 and each pins one
        # action; with the neutral option gone (2026-09-06) no selection is free
        # to choose either, so both contradictions are refused.
        high_send = await tool.execute(
            setting="High",
            outreach_action="send",
            outreach_message="A deadline changed.",
        )
        low_suppress = await tool.execute(
            setting="Low",
            outreach_action="suppress",
            outreach_message="A deadline changed.",
        )

        assert "Low requires outreach_action='suppress'" in low_send
        assert "High requires outreach_action='send'" in high_suppress
        assert '"outreach_action": "send"' in high_send
        assert '"outreach_action": "suppress"' in low_suppress

    asyncio.run(run())


def test_ix_tool_descriptions_preserve_operational_boundaries():
    descriptions = {
        attribute: spec.setting_descriptions
        for attribute, spec in IX_PASSIVE_SPECS.items()
    }

    assert "at most one supporting move" in descriptions["verbosity"]["Terse"]
    assert "Complete the task" in descriptions["emotional_engagement"]["Task-focused"]
    assert "user-facing rationale" in descriptions["reasoning_visibility"]["Show"]
    assert "materially affect action, safety, or outcome" in descriptions["uncertainty_expression"]["Hide"]
    assert "every `_ix_notice` in call order" in descriptions["process_visibility"]["Full narration"]
    assert "without prior confirmation" in descriptions["autonomy_level"]["Self-directed"]
    assert "compare the tradeoffs that separate them" in descriptions["solution_breadth"]["High"]
    assert "carry out no adjacent step" in descriptions["task_expansion"]["Low"]
    assert "does not make you High" in descriptions["task_expansion"]["Low"]
    assert "does not count as the adjacent step" in descriptions["task_expansion"]["High"]
    assert "clearly partitioned set" in descriptions["information_elicitation"]["Structured"]
    assert "explicitly retain the rest" in descriptions["topic_management"]["One-at-a-time"]


def test_the_certainty_block_is_asked_for_on_both_carrier_surfaces():
    """Carrier F — the per-turn tool AND the per-turn reminder.

    A carrier that has to survive to the very last line of the answer is asked
    for twice: once where the payload used to live, once in the reminder that
    is injected before the first provider request of every turn. Both texts are
    setting-blind, so their presence tells the model nothing about the setting.
    """
    reminder = turn_reminder_message([], ["uncertainty_expression"])

    assert CERTAINTY_BLOCK_INSTRUCTION in reminder
    assert CERTAINTY_BLOCK_INSTRUCTION in (
        IX_RENDER_TOOL_SPECS["uncertainty_expression"].description
    )
    for setting in IX_PASSIVE_SPECS["uncertainty_expression"].settings:
        assert setting not in reminder
    # Another session's attributes do not drag the block instruction in.
    assert turn_reminder_message([], ["topic_management"]) == ""
    assert turn_reminder_message([]) == ""


def test_the_pre_turn_directive_says_where_a_claim_is_copied_from():
    """regress14b acc_035 — the claims were sentences of the *previous* reply.

    The requirement is in `CERTAINTY_BLOCK_INSTRUCTION` and in the repair
    message, but the repair only goes out after the turn has already missed. It
    is restated on its own line before the first attempt, on the turn that
    selects the setting and on every later turn that inherits it. Still
    setting-blind: naming no setting, it reads the same under either one.
    """
    express = [IXSelection(
        attribute="uncertainty_expression",
        tool_name="IX_uncertainty_expression",
        setting="Express",
    )]

    for selections in ([], express):
        reminder = turn_reminder_message(selections, ["uncertainty_expression"])
        assert CERTAINTY_CLAIM_SOURCE_REMINDER in reminder
        assert "copied verbatim from the reply you are writing now" in reminder
        assert "never from an earlier reply" in reminder
        for setting in IX_PASSIVE_SPECS["uncertainty_expression"].settings:
            assert setting not in reminder


def test_ix_setting_descriptions_say_what_the_renderer_does():
    """Since 2026-09-05 there is one text per attribute x setting, and
    for a rendered attribute it describes the shape after the renderer, not advice."""
    descriptions = {
        attribute: spec.setting_descriptions
        for attribute, spec in IX_PASSIVE_SPECS.items()
    }

    # what the user ends up seeing, per setting
    assert "prints the paragraph" in descriptions["reasoning_visibility"]["Show"]
    assert "drops that paragraph" in descriptions["reasoning_visibility"]["Hide"]
    assert "first `_ix_notice`, then the result, then your" in descriptions["process_visibility"]["Bookend"]
    assert "no `_ix_notice` is printed" in descriptions["process_visibility"]["Silent"]
    assert "to each claim you list" in descriptions["uncertainty_expression"]["Express"]
    assert "no parenthetical at all" in descriptions["uncertainty_expression"]["Hide"]
    assert "Only the first `[[batch:...]]` block" in descriptions["topic_management"]["One-at-a-time"]
    assert "released at once" in descriptions["information_elicitation"]["Structured"]

    # the gate boundaries the wrapper actually enforces
    assert "every draft, internal write and external action" in descriptions["autonomy_level"]["Reactive"]
    assert "never stops the turn even if you mark" in descriptions["autonomy_level"]["Self-directed"]
    assert "always stops the turn" in descriptions["autonomy_level"]["Self-directed"]
    # ... and what earns `is_key`, since that is the only thing standing between
    # a Self-directed selection and the gate. Live `acc_039` marked an
    # instructed calendar write key, so the gate turned Self-directed into an
    # ask-before-writing turn while the selection value still matched.
    self_directed = descriptions["autonomy_level"]["Self-directed"]
    assert "irreversible, by affecting another person, or by spending money" in self_directed
    assert "a write the user just instructed, on the user's own records" in self_directed
    assert "another write could undo is not key" in self_directed
    # ... and the one the wrapper explicitly does not enforce
    assert "No gate at all" in descriptions["autonomy_level"]["Autonomous"]
    assert "does not hold this line for you" in descriptions["autonomy_level"]["Autonomous"]

    # every setting text is written out, not a one-liner
    for attribute, per_setting in descriptions.items():
        for setting, text in per_setting.items():
            assert len(text) >= 110, f"{attribute}/{setting} description is too thin"


def test_wrapper_notice_reaches_the_system_prompt():
    """The model is told a renderer selects and orders but never rewrites."""
    from pa.agent.ix.specs import WRAPPER_NOTICE

    with tempfile.TemporaryDirectory() as td:
        prompt = ContextBuilder(Path(td)).build_system_prompt()

    assert WRAPPER_NOTICE in prompt
    assert "never rewrites your wording" in prompt


def test_the_prompt_says_a_withdrawn_selection_tool_is_not_called_again():
    """Owner ruling 2026-09-08 (regress20 acc_027).

    The PA re-sent `IX_emotional_engagement` on the beat after the one that
    offered it, read the registry's "not found" as a broken call, and re-sent it
    twice more. The wrapper now ignores the call; the prompt says not to make it.
    """
    with tempfile.TemporaryDirectory() as td:
        prompt = ContextBuilder(Path(td)).build_system_prompt()

    assert (
        "Only tools present in the current turn's tool list may be called; a "
        "selection tool that is no longer listed must not be called again."
    ) in prompt


def test_a_readiness_question_is_answered_with_readiness():
    """acc_027 b5 — "Are you prepared to transmit …?" was answered by sending.

    Marcus got a premature text and, once the next beat instructed the real
    send, a second superseding one seconds later. The rule ships in the base
    guidelines rather than on `autonomy_level`, because that session cued no
    autonomy attribute and the floor never armed.
    """
    from pa.agent.ix.specs import READINESS_QUESTION_NOTE

    with tempfile.TemporaryDirectory() as td:
        prompt = ContextBuilder(Path(td)).build_system_prompt()

    assert READINESS_QUESTION_NOTE in prompt
    assert "ready, able or prepared" in READINESS_QUESTION_NOTE
    assert "is a question, not an instruction to do it" in READINESS_QUESTION_NOTE
    assert "wait for the go-ahead" in READINESS_QUESTION_NOTE


def test_an_outward_message_needs_an_instruction_naming_it():
    """acc_027 t=13 — "book that slot" was answered with a booking and a text.

    Marcus got three messages in a session whose draft specifies one, and the
    middle one was uninstructed. Third round, third route: the readiness rule
    above closed the misread-question route and this closes over-eagerness off
    a task order. Base guidelines again, because that session cued no
    `autonomy_level` and the `external_action_floor` never armed.
    """
    from pa.agent.ix.specs import OUTWARD_MESSAGE_NOTE

    with tempfile.TemporaryDirectory() as td:
        prompt = ContextBuilder(Path(td)).build_system_prompt()

    assert OUTWARD_MESSAGE_NOTE in prompt
    assert "`message_send`" in OUTWARD_MESSAGE_NOTE
    assert "`email_message_send`" in OUTWARD_MESSAGE_NOTE
    assert "names that message" in OUTWARD_MESSAGE_NOTE
    assert "is not licence to notify anyone" in OUTWARD_MESSAGE_NOTE


def test_memory_store():
    """MarkdownMemoryStore reads and writes memory files."""
    with tempfile.TemporaryDirectory() as td:
        ws = Path(td)
        store = MarkdownMemoryStore(ws, log_scope="test")

        assert store.read_long_term() == ""
        store.write_long_term("User prefers formal language.")
        assert "formal" in store.read_long_term()

        store.append_history("[2026-04-01] Greeted user.")
        assert store.history_file.exists()

        ctx = store.get_memory_context()
        assert "Long-term Memory" in ctx

    print("OK: MarkdownMemoryStore works")


if __name__ == "__main__":
    test_all_imports()
    test_config_trimmed()
    test_context_benchmark_identity()
    test_session_checkpoint()
    test_tool_registry()
    test_turn_tool_registry_exposes_active_ix_only()
    test_memory_store()
    print("\n=== All structural tests passed ===")
