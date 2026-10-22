"""Contract tests for the benchmark MCP configuration schema."""

import pytest
from pydantic import ValidationError

from pa.config import MCPServerConfig as ExportedMCPServerConfig
from pa.config.schema import MCPServerConfig


def test_mcp_server_config_export_and_defaults_are_stable():
    config = MCPServerConfig()

    assert ExportedMCPServerConfig is MCPServerConfig
    assert list(MCPServerConfig.model_fields) == [
        "type",
        "command",
        "args",
        "env",
        "url",
        "headers",
        "tool_timeout",
        "enabled_tools",
    ]
    assert config.model_dump() == {
        "type": None,
        "command": "",
        "args": [],
        "env": {},
        "url": "",
        "headers": {},
        "tool_timeout": 30,
        "enabled_tools": ["*"],
    }

    other = MCPServerConfig()
    assert config.args is not other.args
    assert config.env is not other.env
    assert config.headers is not other.headers
    assert config.enabled_tools is not other.enabled_tools


def test_mcp_server_config_accepts_complete_snake_case_input_verbatim():
    payload = {
        "type": "streamableHttp",
        "command": "uv",
        "args": ["run", "state-server"],
        "env": {"ExternalMixedCase": "kept", "MEMPA_STATE_ROOT": "/state"},
        "url": "https://mcp.example.test/api",
        "headers": {"X-Custom-Header": "kept", "Authorization": "secret"},
        "tool_timeout": 45,
        "enabled_tools": ["document_read", "mcp_state_email_draft_create"],
    }

    assert MCPServerConfig.model_validate(payload).model_dump() == payload


@pytest.mark.parametrize(
    "payload",
    [
        {"toolTimeout": 45},
        {"enabledTools": ["document_read"]},
        {"tool_timout": 45},
    ],
)
def test_mcp_server_config_rejects_legacy_and_unknown_fields(payload):
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        MCPServerConfig.model_validate(payload)


def test_mcp_server_config_rejects_unknown_transport_value():
    with pytest.raises(ValidationError):
        MCPServerConfig(type="streamable_http")
