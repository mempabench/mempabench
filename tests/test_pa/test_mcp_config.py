"""Tests for MCP configuration-facing runtime behavior."""

import asyncio
from contextlib import AsyncExitStack, asynccontextmanager
from types import SimpleNamespace

import mcp
import mcp.client.stdio

from pa.agent.tools import mcp as mcp_tools
from pa.agent.tools.registry import ToolRegistry
from pa.config.schema import MCPServerConfig


class _FakeClientSession:
    def __init__(self, _read, _write):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, _exc_type, _exc, _traceback):
        return False

    async def initialize(self):
        return None

    async def list_tools(self):
        return SimpleNamespace(
            tools=[
                SimpleNamespace(
                    name="document_read",
                    description="Read a document",
                    inputSchema={"type": "object", "properties": {}},
                )
            ]
        )


@asynccontextmanager
async def _fake_stdio_client(_params):
    yield object(), object()


def test_enabled_tools_logs_use_the_real_snake_case_field_name(monkeypatch):
    monkeypatch.setattr(mcp, "ClientSession", _FakeClientSession)
    monkeypatch.setattr(
        mcp,
        "StdioServerParameters",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )
    monkeypatch.setattr(mcp.client.stdio, "stdio_client", _fake_stdio_client)

    messages: list[str] = []
    sink_id = mcp_tools.logger.add(
        lambda message: messages.append(str(message)),
        format="{message}",
        level="DEBUG",
    )

    async def run() -> None:
        config = MCPServerConfig(
            type="stdio",
            command="fake-server",
            enabled_tools=["missing_tool"],
        )
        async with AsyncExitStack() as stack:
            await mcp_tools.connect_mcp_servers(
                {"state": config}, ToolRegistry(), stack
            )

    try:
        asyncio.run(run())
    finally:
        mcp_tools.logger.remove(sink_id)

    combined = "\n".join(messages)
    assert "not in enabled_tools" in combined
    assert "enabled_tools entries not found" in combined
    assert "enabledTools" not in combined
