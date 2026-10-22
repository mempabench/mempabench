"""A spent forcing hint is not evidence of three unresolved IX attempts."""
import asyncio
import socket
import pytest
from test_ix_operation_repair_budget import ScriptedProvider, RecordingTask, tool
from pa.agent.runner import AgentRunSpec, AgentRunner
from pa.agent.ix import TurnToolRegistry
from pa.agent.tools.registry import ToolRegistry
from pa.providers.base import LLMResponse

@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    monkeypatch.setattr(socket.socket, 'connect', lambda *a, **k: (_ for _ in ()).throw(AssertionError('offline')))

def run_case(*, succeeds=False, length=False, business=True):
    selection = tool('IX_reasoning_visibility', setting='Show', evidence='Request.', application='Explain.')
    responses = [
        LLMResponse(content=None, tool_calls=[selection]),
        LLMResponse(content='First missing render.'),
        LLMResponse(content='Second missing render.'),
        # The forced render is delivered but the existing editorial nudge asks
        # for a reader-facing rationale before banking any of this response.
        LLMResponse(content=None, tool_calls=[tool('IX_render_reasoning_visibility', user_facing_rationale='The user approved this action.')]),
        # A valid task executes exactly once; the selection is already banked.
        LLMResponse(content=None, tool_calls=[selection, *([tool('task_write', value='once')] if business else [])]),
    ]
    if succeeds:
        responses.extend([LLMResponse(content=None, tool_calls=[tool('IX_render_reasoning_visibility', user_facing_rationale='This helps you finish.')]), LLMResponse(content='Done.')])
    else:
        responses.extend([LLMResponse(content='Third actual missing render.', finish_reason='length' if length else 'stop'), LLMResponse(content='Must not request a fourth miss.')])
    provider=ScriptedProvider(responses);writes=[];base=ToolRegistry();base.register(RecordingTask(writes))
    context={'session_active_ix':['reasoning_visibility'], 'tool_action_types':{'task_write':'internal_write'}}
    tools=TurnToolRegistry(base,['reasoning_visibility'],ix_context=context)
    result=asyncio.run(AgentRunner(provider).run(AgentRunSpec(initial_messages=[{'role':'user','content':'Same ACC context.'}],tools=tools,model='offline',max_iterations=16,required_tool_names=tools.required_ix_tool_names,ix_context=context)))
    return result,provider,writes

@pytest.mark.parametrize('business',[False,True])
@pytest.mark.parametrize('succeeds',[False,True])
@pytest.mark.parametrize('length',[False,True])
def test_nudge_and_duplicate_selection_do_not_reset_outstanding_render_budget(succeeds,length,business):
    result,provider,writes=run_case(succeeds=succeeds,length=length,business=business)
    assert writes==(['once'] if business else [])
    assert sum(e.get('name')=='ix_rationale_work_log' for e in result.tool_events)==1
    forced=[e for e in result.tool_events if e.get('name')=='ix_render_payload_forced']
    assert [e['tool'] for e in forced]==['IX_render_reasoning_visibility']
    assert sum(isinstance(c,dict) and c.get('function',{}).get('name')=='IX_render_reasoning_visibility' for c in provider.choices)==1
    assert all(c[0]==provider.calls[0][0] for c in provider.calls)
    if succeeds:
        assert result.stop_reason=='completed'
        assert result.ix_contract_exhaustion is None
        assert len(provider.calls)==7 and not provider.responses
    else:
        assert result.stop_reason=='ix_contract_failure'
        assert result.ix_contract_exhaustion=={'attempts':3,'contract':'render','operation':'required_render_carrier'}
        assert len(provider.calls)==6 and len(provider.responses)==1


@pytest.mark.parametrize('succeeds',[False,True])
def test_unavailable_force_hint_is_not_early_exhaustion(succeeds):
    # A real provider policy without forced choices reaches the same fallback
    # as an exhausted pin. It still owes a third response, not an early fatal.
    responses=[LLMResponse(content=None,tool_calls=[tool('IX_reasoning_visibility',setting='Show',evidence='Request.',application='Explain.')]),LLMResponse(content='Missing one.'),LLMResponse(content='Missing two.')]
    if succeeds:
        responses.extend([LLMResponse(content=None,tool_calls=[tool('IX_render_reasoning_visibility',user_facing_rationale='This helps you.')]),LLMResponse(content='Done.')])
    else:
        responses.extend([LLMResponse(content='Missing three.'),LLMResponse(content='No fourth.')])
    provider=ScriptedProvider(responses)
    provider.supports_forced_tool_choice=lambda model:False
    tools=TurnToolRegistry(ToolRegistry(),['reasoning_visibility'],ix_context={'session_active_ix':['reasoning_visibility']})
    result=asyncio.run(AgentRunner(provider).run(AgentRunSpec(initial_messages=[{'role':'user','content':'Same context.'}],tools=tools,model='offline',max_iterations=12,required_tool_names=tools.required_ix_tool_names,ix_context={'session_active_ix':['reasoning_visibility']})))
    assert not any(e.get('name')=='ix_render_payload_forced' for e in result.tool_events)
    assert any(e.get('name')=='ix_render_payload_missing' for e in result.tool_events)
    if succeeds:
        assert result.stop_reason=='completed' and len(provider.calls)==5
    else:
        assert result.ix_contract_exhaustion=={'attempts':3,'contract':'render','operation':'required_render_carrier'}
        assert len(provider.calls)==4 and len(provider.responses)==1
