from typing import Any

import pytest

from pa.agent.loop import AgentLoop
from pa.agent.tools.base import Tool
from pa.agent.tools.registry import ToolRegistry
from pa.agent.runner import AgentRunner, AgentRunSpec
from pa.providers.base import LLMProvider, LLMResponse, ToolCallRequest


class _ExplodingTool(Tool):
    @property
    def name(self) -> str:
        return "explode"

    @property
    def description(self) -> str:
        return "Raise an internal failure."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "count": {"type": "integer", "minimum": 1},
                "api_key": {"type": "integer"},
            },
            "required": ["count"],
        }

    async def execute(self, **kwargs: Any) -> Any:
        raise RuntimeError(f"internal failure for count={kwargs['count']!r}")


class _ToolCallingProvider(LLMProvider):
    def get_default_model(self) -> str:
        return "test-model"

    async def chat(self, *args: Any, **kwargs: Any) -> LLMResponse:
        return LLMResponse(
            content=None,
            tool_calls=[
                ToolCallRequest(
                    id="call_1",
                    name="explode",
                    arguments={"count": 2},
                )
            ],
        )


@pytest.mark.asyncio
async def test_unknown_tool_execution_error_propagates():
    registry = ToolRegistry()
    registry.register(_ExplodingTool())

    with pytest.raises(RuntimeError, match="count=2"):
        await registry.execute("explode", {"count": 2})


@pytest.mark.asyncio
async def test_agent_runner_does_not_convert_unknown_tool_error_to_result():
    registry = ToolRegistry()
    registry.register(_ExplodingTool())

    with pytest.raises(RuntimeError, match="count=2"):
        await AgentRunner(_ToolCallingProvider()).run(
            AgentRunSpec(
                initial_messages=[{"role": "user", "content": "run it"}],
                tools=registry,
                model="test-model",
                max_iterations=2,
            )
        )


@pytest.mark.asyncio
async def test_agent_loop_propagates_mcp_connection_failure(monkeypatch):
    loop = AgentLoop.__new__(AgentLoop)
    loop._mcp_connected = False
    loop._mcp_connecting = False
    loop._mcp_servers = {"state": {}}
    loop._mcp_stack = None
    loop.tools = ToolRegistry()

    async def fail_connection(*args, **kwargs):
        raise RuntimeError("MCP connection failed at source")

    monkeypatch.setattr(
        "pa.agent.tools.mcp.connect_mcp_servers",
        fail_connection,
    )

    with pytest.raises(RuntimeError, match="MCP connection failed at source"):
        await loop._connect_mcp()

    assert loop._mcp_connected is False
    assert loop._mcp_connecting is False
    assert loop._mcp_stack is None


@pytest.mark.asyncio
async def test_invalid_tool_input_reports_value_without_executing():
    registry = ToolRegistry()
    registry.register(_ExplodingTool())

    result = await registry.execute("explode", {"count": 0})

    assert "count must be >= 1; value=0" in result


@pytest.mark.asyncio
async def test_sensitive_invalid_tool_input_is_redacted():
    registry = ToolRegistry()
    registry.register(_ExplodingTool())

    result = await registry.execute(
        "explode",
        {"count": 1, "api_key": "do-not-print-this"},
    )

    assert "api_key should be integer; value=<redacted>" in result
    assert "do-not-print-this" not in result
