"""Model execution loop for the simulator agent.

Deliberate COPY of `pa/agent/runner.py` as of commit d396375, not a shared
import. `pa/` and `simulator/` never share modules: the PA under test and the
simulated user must be able to drift independently, and `harness/` is the only
integration point. Sharing this loop would have made every PA-side change to
tool policy or IX enforcement silently change the user simulator too.

Trimmed to what the simulator actually uses. Dropped from the PA version: tool
execution and the IX-first ordering, autonomy/elicitation gating and repair
prompts, process-visibility and response-directive events, hooks and streaming,
`llm_task_identity`, and the PA tool-result error classification. The simulator
is a user actor: it registers no tools and runs a single iteration, so those
branches were unreachable here. Public names (`AgentRunner`, `AgentRunSpec`,
`AgentRunResult`) are kept identical to the PA version on purpose.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from simulator.providers.base import LLMProvider

_DEFAULT_MAX_ITERATIONS_MESSAGE = (
    "I reached the maximum number of tool call iterations ({max_iterations}) "
    "without completing the task. You can try breaking the task into smaller steps."
)
_DEFAULT_ERROR_MESSAGE = "Sorry, I encountered an error calling the AI model."


@dataclass(slots=True)
class AgentRunSpec:
    """Configuration for a single agent execution."""

    initial_messages: list[dict[str, Any]]
    tools: Any
    model: str
    max_iterations: int
    temperature: float | None = None
    max_tokens: int | None = None
    reasoning_effort: str | None = None
    error_message: str | None = _DEFAULT_ERROR_MESSAGE
    max_iterations_message: str | None = None


@dataclass(slots=True)
class AgentRunResult:
    """Outcome of a simulator agent execution."""

    final_content: str | None
    messages: list[dict[str, Any]]
    usage: dict[str, int] = field(default_factory=dict)
    stop_reason: str = "completed"
    error: str | None = None


class AgentRunner:
    """Run the simulator's LLM loop. The simulator calls no tools."""

    def __init__(self, provider: LLMProvider):
        self.provider = provider

    async def run(self, spec: AgentRunSpec) -> AgentRunResult:
        messages = list(spec.initial_messages)
        final_content: str | None = None
        usage = {"prompt_tokens": 0, "completion_tokens": 0}
        error: str | None = None
        stop_reason = "completed"

        for _iteration in range(spec.max_iterations):
            provider_request: dict[str, Any] = {
                "messages": messages,
                "tools": spec.tools.get_definitions(),
                "model": spec.model,
            }
            if spec.temperature is not None:
                provider_request["temperature"] = spec.temperature
            if spec.max_tokens is not None:
                provider_request["max_tokens"] = spec.max_tokens
            if spec.reasoning_effort is not None:
                provider_request["reasoning_effort"] = spec.reasoning_effort

            provider_response = await self.provider.chat_with_retry(**provider_request)

            raw_usage = provider_response.usage or {}
            usage = {
                "prompt_tokens": int(raw_usage.get("prompt_tokens", 0) or 0),
                "completion_tokens": int(raw_usage.get("completion_tokens", 0) or 0),
            }

            if provider_response.has_tool_calls:
                # The simulator registers no tools, so a tool call here means
                # the model invented one. The PA version would execute it.
                #
                # Reported, not raised: the raise happened before the agent
                # loop's output-contract retry, so one hallucinated call killed
                # the whole orchestrator process (live memos `acc_034`, gemini).
                # A tool call is a message that cannot be used as it stands,
                # which is what `simulator_output_format_invalid` already means,
                # so the loop turns this `stop_reason` into one rejected
                # attempt, retries within the existing budget, and voids the
                # session the normal way if every attempt does it.
                stop_reason = "tool_call"
                error = (
                    "simulator agent received tool calls but has no tools: "
                    + ", ".join(
                        tool_call.name for tool_call in provider_response.tool_calls
                    )
                )
                final_content = provider_response.content or ""
                break

            content = provider_response.content
            if provider_response.finish_reason == "error":
                final_content = (
                    content or spec.error_message or _DEFAULT_ERROR_MESSAGE
                )
                stop_reason = "error"
                error = final_content
                break

            messages.append({"role": "assistant", "content": content})
            final_content = content
            break
        else:
            stop_reason = "max_iterations"
            template = spec.max_iterations_message or _DEFAULT_MAX_ITERATIONS_MESSAGE
            final_content = template.format(max_iterations=spec.max_iterations)

        return AgentRunResult(
            final_content=final_content,
            messages=messages,
            usage=usage,
            stop_reason=stop_reason,
            error=error,
        )
