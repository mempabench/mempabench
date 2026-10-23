from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from simulator.memory import MemoryConsolidationError
from simulator.profiles import load_simulator_profile
from simulator.providers.base import LLMProvider, LLMResponse, ToolCallRequest


class _MemoryProvider(LLMProvider):
    def get_default_model(self) -> str:
        return "memory-model"

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> LLMResponse:
        return LLMResponse(
            content=None,
            tool_calls=[
                ToolCallRequest(
                    id="call_save_memory",
                    name="save_memory",
                    arguments={
                        "history_entry": "[2026-05-17 12:00] Test User answered a greeting.",
                        "memory_update": "- Test User has answered a greeting.",
                    },
                )
            ],
        )


def test_load_starter_profile_reads_nanobot_workspace() -> None:
    profile = load_simulator_profile("simulator/profiles/user_a")

    assert profile.id == "user_a"
    assert profile.display_name == "User A"
    assert profile.memory_mode == "nanobot"
    assert profile.identity_path == Path("data/personas/user_a_identity.yaml").resolve()
    assert profile.preference_matrix_path == Path("data/personas/user_a_preferences.yaml").resolve()
    assert profile.actor_skill_specs_root == Path("data/actor_skill_specs").resolve()

    context = profile.build_context()

    assert "## Identity" in context
    assert "## Ground rules" in context
    assert "## TOOLS.md" in context
    assert "## SOUL.md" not in context
    assert "## USER.md" not in context


@pytest.mark.parametrize(
    ("profile_id", "background", "trait", "sentence_pattern", "act"),
    [
        (
            "user_a",
            "Theoretical physicist at Westlake Institute of Technology",
            "Obsessively precise",
            "Declarative assertions without hedging",
            "Demand precise specifications before proceeding",
        ),
        # Restored with the strings the re-extracted bundle
        # actually carries; the old ones came from the hand-written user_b that
        # was archived before the extraction pipeline built this identity.
        (
            "user_b",
            "Waitress/bartender and struggling actress at The Garden Bistro",
            "She is insecure about her intelligence and formal education",
            "Asks rhetorical questions to express disbelief, challenge logic, or mock a premise",
            "She offers practical solutions or simple alternatives when others overcomplicate problems",
        ),
    ],
)
def test_build_context_renders_the_identity_yaml(
    profile_id: str,
    background: str,
    trait: str,
    sentence_pattern: str,
    act: str,
) -> None:
    """The persona content the deleted SOUL.md/USER.md carried by hand now
    reaches the prompt from `data/personas/<id>_identity.yaml`."""
    profile = load_simulator_profile(f"simulator/profiles/{profile_id}")
    context = profile.build_context()

    assert profile.display_name in context
    assert background in context
    assert trait in context
    assert sentence_pattern in context
    assert act in context
    # Persona-independent framing, now rendered from simulator/prompts/.
    assert (
        f"You are {profile.display_name}. You are the person speaking to a "
        "personal assistant in this interaction." in context
    )
    assert "Use the session facts as the active task ground truth." in context
    # The tool boundary policy still comes from the profile workspace file.
    assert "You are a simulated user, not the assistant under test." in context
    assert "## SOUL.md" not in context
    assert "## USER.md" not in context


def test_profile_records_session_with_user_memory(tmp_path: Path) -> None:
    import asyncio

    root = tmp_path / "test_user"
    (root / "memory").mkdir(parents=True)
    profile = load_simulator_profile(root)

    assert profile.id == "test_user"
    assert profile.display_name == "Test User"

    asyncio.run(
        profile.record_session(
            [
                {"role": "pa", "content": "hello"},
                {"role": "simulated_user", "content": "I am Test User."},
            ],
            provider=_MemoryProvider(),
            model="memory-model",
            session_id="session_test",
            timestamp="2026-10-01 09:00",
        )
    )

    history = (root / "memory" / "HISTORY.md").read_text(encoding="utf-8")
    assert "Test User answered a greeting" in history
    memory = (root / "memory" / "MEMORY.md").read_text(encoding="utf-8")
    assert "answered a greeting" in memory


def test_user_memory_prompt_is_first_person_and_excludes_domain_content() -> None:
    from simulator.memory import render_user_memory_prompt

    prompt = render_user_memory_prompt(
        user_name="Test User",
        timestamp="2026-09-03 21:16",
        current_memory="",
        exchanges=[
            {"role": "pa", "content": "I looked in the wrong folder first."},
            {"role": "simulated_user", "content": "Try the research folder."},
        ],
    )

    # First-person recollection, not a third-person report about a user.
    assert "You are Test User." in prompt
    assert "Write in the first person" in prompt
    assert "not a research notebook" in prompt

    # The experience of working with the assistant is what gets recorded.
    for rule in (
        "what I asked the assistant to do",
        "did I have to correct it",
        "what I decided or authorized",
        "what is still open",
    ):
        assert rule in prompt

    # The exclusion list.
    for excluded in (
        "the subject matter itself",
        "file paths, folder names, file names",
        "exact numbers, dates, quantities, citations",
        "lists of caveats, limits, conditions, or principles",
    ):
        assert excluded in prompt

    # One entry per session, short.
    assert "one entry for this whole session, 2 to 4 sentences" in prompt

    # Transcript is rendered from the user's point of view.
    assert "Assistant: I looked in the wrong folder first." in prompt
    assert "Me: Try the research folder." in prompt
    assert "(empty)" in prompt


class _BudgetRecordingProvider(LLMProvider):
    """Replays a scripted finish_reason per call, recording the budget it got.

    "ok" returns a valid save_memory call; anything else returns a reply with
    no tool call and that finish_reason, which is what the store turns into a
    contract error. "error" is avoided on purpose: `chat_with_retry` sleeps on
    it, and none of these cases is about transient provider failure.
    """

    def __init__(self, finish_reasons: list[str]) -> None:
        super().__init__()
        self.finish_reasons = list(finish_reasons)
        self.max_tokens_seen: list[int] = []
        self.prompts: list[str] = []

    def get_default_model(self) -> str:
        return "memory-model"

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> LLMResponse:
        self.max_tokens_seen.append(max_tokens)
        self.prompts.append(messages[0]["content"])
        reason = self.finish_reasons.pop(0)
        if reason == "ok":
            return LLMResponse(
                content=None,
                tool_calls=[
                    ToolCallRequest(
                        id="call_save_memory",
                        name="save_memory",
                        arguments={
                            "history_entry": "[2026-10-01 09:00] Test User said hello.",
                            "memory_update": "- Test User says hello.",
                        },
                    )
                ],
            )
        return LLMResponse(content="", finish_reason=reason)


def _record(profile, provider) -> None:
    import asyncio

    asyncio.run(
        profile.record_session(
            [
                {"role": "pa", "content": "hello"},
                {"role": "simulated_user", "content": "I am Test User."},
            ],
            provider=provider,
            model="memory-model",
            session_id="acc_107",
            timestamp="2026-10-01 09:00",
        )
    )


def _blank_profile(tmp_path: Path):
    root = tmp_path / "test_user"
    (root / "memory").mkdir(parents=True)
    return load_simulator_profile(root)


def test_memory_write_retries_a_cut_off_reply_at_twice_the_budget(
    tmp_path: Path,
) -> None:
    """A reply cut at the cap is re-asked wider, and the entry still lands."""
    profile = _blank_profile(tmp_path)
    provider = _BudgetRecordingProvider(["length", "ok"])

    _record(profile, provider)

    # The provider default, then double it -- no third call.
    assert provider.max_tokens_seen == [4096, 8192]
    history = (tmp_path / "test_user" / "memory" / "HISTORY.md").read_text(
        encoding="utf-8"
    )
    assert "Test User said hello" in history


def test_memory_write_failure_carries_the_budget_history(tmp_path: Path) -> None:
    """Three cut replies: still 3 attempts, and the reason names every budget."""
    profile = _blank_profile(tmp_path)
    provider = _BudgetRecordingProvider(["length", "length", "length"])

    with pytest.raises(MemoryConsolidationError) as excinfo:
        _record(profile, provider)

    # The retry grows the existing attempts; it does not add a fourth call.
    assert provider.max_tokens_seen == [4096, 8192, 16384]
    message = str(excinfo.value)
    assert "INVALID_CONSOLIDATION" in message
    assert "failed after 3 attempts" in message
    assert "finish_reason=length x3" in message
    assert "max_tokens=4096→8192→16384" in message
    assert "session_id='acc_107'" in message
    # The session is not voided: the caller logs this and carries on.
    assert not (tmp_path / "test_user" / "memory" / "HISTORY.md").exists()


def test_memory_write_keeps_the_budget_on_a_non_length_failure(
    tmp_path: Path,
) -> None:
    """Only truncation grows the budget; a broken contract is re-asked as is."""
    profile = _blank_profile(tmp_path)
    provider = _BudgetRecordingProvider(["stop", "stop", "stop"])

    with pytest.raises(MemoryConsolidationError) as excinfo:
        _record(profile, provider)

    assert provider.max_tokens_seen == [4096, 4096, 4096]
    message = str(excinfo.value)
    assert "finish_reason=stop x3" in message
    assert "max_tokens=4096→4096→4096" in message


def test_cut_off_retry_asks_for_a_shorter_note(tmp_path: Path) -> None:
    """Prompt lock: the re-ask after truncation carries the shorter-note ask."""
    from simulator.memory import SHORTER_NOTE_INSTRUCTION

    profile = _blank_profile(tmp_path)
    provider = _BudgetRecordingProvider(["length", "ok"])

    _record(profile, provider)

    first, retry = provider.prompts
    assert SHORTER_NOTE_INSTRUCTION not in first
    assert SHORTER_NOTE_INSTRUCTION in retry
    # The instruction itself, locked like the other prompt text.
    assert "ran out of output space before the save_memory call" in retry
    assert "hold history_entry to" in retry
    assert "2 sentences" in retry
    # Nothing else about the prompt changed on the re-ask.
    assert retry == first + SHORTER_NOTE_INSTRUCTION


def test_non_length_retry_leaves_the_prompt_untouched(tmp_path: Path) -> None:
    """The shorter-note ask rides with the budget, not with every retry."""
    from simulator.memory import SHORTER_NOTE_INSTRUCTION

    profile = _blank_profile(tmp_path)
    provider = _BudgetRecordingProvider(["stop", "ok"])

    _record(profile, provider)

    first, retry = provider.prompts
    assert SHORTER_NOTE_INSTRUCTION not in first
    assert retry == first
