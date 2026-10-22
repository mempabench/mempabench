"""Keep provider JSON Schema and MCP's actual optional/null contract aligned."""
import asyncio
from copy import deepcopy
import json
import socket
from types import SimpleNamespace

from jsonschema import Draft202012Validator
import pytest

from pa.agent.tools.mcp import MCPToolWrapper
from pa.agent.tools.registry import ToolRegistry
from pa.providers.openai_compat_provider import OpenAICompatProvider
from state_server.server import mcp


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def deny(*args, **kwargs):
        raise AssertionError('network forbidden')
    monkeypatch.setattr(socket.socket, 'connect', deny)


def wrap(schema, name='example', session=None):
    return MCPToolWrapper(session, 'state', SimpleNamespace(
        name=name, description='offline', inputSchema=schema))


@pytest.mark.parametrize('name,key,base', [
    ('email_draft_create', 'draft_id', {'to': 'offline', 'subject': 'x', 'body': 'x'}),
    ('payment_method_read', 'method_id', {}),
])
@pytest.mark.parametrize('value', ['OMITTED', None, '', '. ', '/', 'draft_0000'])
def test_actual_optional_ids_reach_mcp_unchanged(name, key, base, value):
    actual = mcp._tool_manager.get_tool(name)
    seen = []
    class Session:
        async def call_tool(self, tool, arguments):
            seen.append((tool, arguments))
            return SimpleNamespace(content=[])
    tool = wrap(actual.parameters, name, Session())
    args = dict(base)
    if value != 'OMITTED':
        args[key] = value
    # Both are valid schema instances; blank/nonexistent IDs are still business
    # errors. Do not "repair" them into create/list operations.
    Draft202012Validator(actual.parameters).validate(args)
    Draft202012Validator(tool.to_schema()['function']['parameters']).validate(args)
    assert key not in tool.parameters.get('required', [])
    provider = object.__new__(OpenAICompatProvider)
    response = provider._parse({'choices': [{'finish_reason': 'tool_calls', 'message': {
        'tool_calls': [{'id': 'offline', 'type': 'function', 'function': {
            'name': tool.name, 'arguments': json.dumps(args)}}]}}]})
    assert response.tool_calls[0].arguments == args
    registry = ToolRegistry()
    registry.register(tool)
    asyncio.run(registry.execute(tool.name, response.tool_calls[0].arguments))
    assert seen == [(name, args)]


@pytest.mark.parametrize('union', [
    {'anyOf': [{'type': 'string'}, {'type': 'null'}]},
    {'oneOf': [{'type': 'integer', 'minimum': 2}, {'type': 'null'}]},
    {'type': ['array', 'null'], 'items': {'type': 'integer'}},
    {'anyOf': [{'type': 'string', 'enum': ['a', 'b']}, {'type': 'null'}]},
])
def test_nullable_union_preserved_and_validated_locally(union):
    schema = {'type': 'object', 'properties': {'value': union}, 'required': ['value']}
    before = deepcopy(schema)
    tool = wrap(schema)
    assert tool.parameters == before
    assert tool.validate_params({'value': None}) == []
    Draft202012Validator(tool.parameters).validate({'value': None})
    assert tool.validate_params({})
    assert schema == before
    # The outbound schema must not share mutable nested dictionaries with MCP.
    tool.parameters['properties']['value']['description'] = 'local edit'
    assert schema == before


def test_local_nested_cast_and_validation_still_apply():
    schema = {'type': 'object', 'properties': {'rows': {
        'anyOf': [{'type': 'array', 'items': {'type': 'object', 'properties': {
            'count': {'anyOf': [{'type': 'integer', 'minimum': 2}, {'type': 'null'}]},
        }, 'required': ['count']}}, {'type': 'null'}]}}, 'required': ['rows']}
    tool = wrap(schema)
    assert tool.cast_params({'rows': [{'count': '3'}]}) == {'rows': [{'count': 3}]}
    assert not tool.validate_params({'rows': [{'count': None}]})
    assert tool.validate_params({'rows': [{'count': 1}]})
    assert tool.validate_params({'rows': [{}]})
    assert tool.validate_params({'rows': 'wrong'})
    assert tool.validate_params([])


@pytest.mark.parametrize('model', [
    'openai/gpt-5.6-sol', 'deepseek/deepseek-v4-flash',
    'qwen/qwen3.8-27b', 'tencent/hy4-preview',
])
def test_provider_request_preserves_standard_nullable_schema(model):
    tool = wrap(mcp._tool_manager.get_tool('payment_method_read').parameters)
    tools = [tool.to_schema()]
    provider = OpenAICompatProvider(api_key='offline-dummy', default_model=model)
    payload = provider._build_kwargs([{'role': 'user', 'content': 'offline'}],
                                    tools, model, 8192, .2, None, None)
    assert payload['tools'] == tools
    actual_schema = payload['tools'][0]['function']['parameters']
    Draft202012Validator(actual_schema).validate({'method_id': None})
    Draft202012Validator(actual_schema).validate({})
    assert 'strict' not in payload['tools'][0]['function']
