"""A resolved selection must not spend a later render operation's repair budget."""
from __future__ import annotations
import asyncio
import socket
from typing import Any
import pytest
from pa.agent.ix import TurnToolRegistry
from pa.agent.runner import AgentRunner, AgentRunSpec
from pa.agent.tools.base import Tool
from pa.agent.tools.registry import ToolRegistry
from pa.providers.base import LLMProvider, LLMResponse, ToolCallRequest

class RecordingTask(Tool):
    def __init__(self, writes): self.writes = writes
    @property
    def name(self): return 'task_write'
    @property
    def description(self): return 'Record a synthetic action once.'
    @property
    def parameters(self): return {'type': 'object', 'properties': {'value': {'type': 'string'}}, 'required': ['value']}
    async def execute(self, value):
        self.writes.append(value)
        return {'ok': True}

class ScriptedProvider(LLMProvider):
    def __init__(self, responses):
        super().__init__(); self.responses = list(responses); self.calls = []; self.choices = []
    def get_default_model(self): return 'offline-test'
    async def chat(self, messages, tools=None, model=None, max_tokens=4096,
                   temperature=0.7, reasoning_effort=None, tool_choice=None):
        self.calls.append(list(messages)); self.choices.append(tool_choice)
        assert self.responses, 'unexpected additional provider attempt'
        return self.responses.pop(0)

@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def denied(*args, **kwargs): raise AssertionError('offline test cannot connect')
    monkeypatch.setattr(socket.socket, 'connect', denied)

def tool(name, **arguments):
    return ToolCallRequest(id='synthetic_' + name, name=name, arguments=arguments)

def run_case(selection_misses, *, final_render_succeeds, business_action=True):
    # The first two responses are voided before any task is executed.
    missed = [LLMResponse(content='Selection omitted.', tool_calls=[
        tool('task_write', value='must-not-execute')]) for _ in range(selection_misses)]
    responses = [*missed, LLMResponse(content=None, tool_calls=[
        tool('IX_reasoning_visibility', setting='Show', evidence='Synthetic request.', application='Give the reason.')])]
    if business_action:
        responses.append(LLMResponse(content=None, tool_calls=[tool('task_write', value='keep-once')]))
    responses.extend([LLMResponse(content='First final response without carrier.'),
                      LLMResponse(content='Second final response without carrier.')])
    if final_render_succeeds:
        responses.extend([LLMResponse(content=None, tool_calls=[
            tool('IX_render_reasoning_visibility', user_facing_rationale='A synthetic reason.')]),
            LLMResponse(content='Completed.')])
    else:
        responses.extend([LLMResponse(content='Third final response without carrier.'),
                          LLMResponse(content='Must not request fourth repair.')])
    provider = ScriptedProvider(responses); writes = []; base = ToolRegistry(); base.register(RecordingTask(writes))
    context = {'session_active_ix': ['reasoning_visibility'], 'tool_action_types': {'task_write': 'internal_write'}}
    tools = TurnToolRegistry(base, ['reasoning_visibility'], ix_context=context)
    result = asyncio.run(AgentRunner(provider).run(AgentRunSpec(
        initial_messages=[{'role': 'user', 'content': 'Same synthetic ACC context.'}],
        tools=tools, model='offline-test', max_iterations=16,
        required_tool_names=tools.required_ix_tool_names, ix_context=context, turn_index=3)))
    return result, provider, writes

@pytest.mark.parametrize('selection_misses', [0, 1, 2])
@pytest.mark.parametrize('business_action', [False, True])
def test_new_render_operation_gets_third_response_after_selection_success(selection_misses, business_action):
    result, provider, writes = run_case(selection_misses, final_render_succeeds=True, business_action=business_action)
    assert result.stop_reason == 'completed'
    assert result.ix_contract_exhaustion is None
    assert result.final_content == 'Completed.'
    assert writes == (['keep-once'] if business_action else [])
    assert not provider.responses
    assert all(call[0] == provider.calls[0][0] for call in provider.calls)
    choices = [v['function']['name'] if isinstance(v, dict) else None for v in provider.choices]
    assert choices[-2:] == ['IX_render_reasoning_visibility', None]

@pytest.mark.parametrize('selection_misses', [0, 1, 2])
@pytest.mark.parametrize('business_action', [False, True])
def test_unresolved_render_stops_at_three_with_typed_evidence(selection_misses, business_action):
    result, provider, writes = run_case(selection_misses, final_render_succeeds=False, business_action=business_action)
    assert result.stop_reason == 'ix_contract_failure'
    assert result.ix_contract_failure_kind == 'render'
    assert result.ix_contract_exhaustion == {'attempts': 3, 'contract': 'render', 'operation': 'required_render_carrier'}
    assert len(provider.responses) == 1 and provider.responses[0].content == 'Must not request fourth repair.'
    assert writes == (['keep-once'] if business_action else [])
    assert not any(e.get('name') == 'tool_retry_cap_reached' for e in result.tool_events)
    assert all(call[0] == provider.calls[0][0] for call in provider.calls)
