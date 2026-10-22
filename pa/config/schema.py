"""Benchmark MCP connection configuration."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class MCPServerConfig(BaseModel):
    """MCP server connection configuration (stdio or HTTP)."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["stdio", "sse", "streamableHttp"] | None = None
    command: str = ""
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    url: str = ""
    headers: dict[str, str] = Field(default_factory=dict)
    tool_timeout: int = 30
    enabled_tools: list[str] = Field(default_factory=lambda: ["*"])
