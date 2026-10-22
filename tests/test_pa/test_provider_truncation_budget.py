"""Empty provider truncations cannot become evidence of IX exhaustion."""
import asyncio
import sys
import socket
import pytest
from pa.agent.ix import TurnToolRegistry
from pa.agent.runner import AgentRunner, AgentRunSpec
from pa.agent.tools.registry import ToolRegistry
from pa.providers.base import LLMResponse
from test_ix_operation_repair_budget import ScriptedProvider, RecordingTask, tool

@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def deny(*a, **k):raise AssertionError('offline test cannot connect')
    monkeypatch.setattr(socket.socket,'connect',deny)
    monkeypatch.setattr(socket.socket,'connect_ex',deny)
    monkeypatch.setattr(socket,'create_connection',deny)

def truncated():
    return LLMResponse(content=None,finish_reason='length',usage={'completion_tokens':8192})

def run_responses(responses, *, ix=True):
    provider=ScriptedProvider(responses);writes=[];base=ToolRegistry();base.register(RecordingTask(writes))
    context={'session_active_ix':['topic_management'],'tool_action_types':{'task_write':'internal_write'}} if ix else {}
    tools=TurnToolRegistry(base,['topic_management'],ix_context=context) if ix else base
    spends=[]
    def trace(frame,event,arg):
        if event=='return' and frame.f_code.co_name=='_spend_ix_repair_attempt':
            spends.append({'request':len(provider.calls),'counter':frame.f_locals['ix_repair_attempts']})
        return trace
    prior=sys.gettrace();sys.settrace(trace)
    try:
        result=asyncio.run(AgentRunner(provider).run(AgentRunSpec(
            initial_messages=[{'role':'user','content':'Same original context.'}],tools=tools,
            model='offline-test',max_iterations=16,ix_context=context,
            required_tool_names=tools.required_ix_tool_names if ix else [],turn_index=3)))
    finally:sys.settrace(prior)
    return result,provider,writes,spends

def missing():
    return LLMResponse(content='Missing IX.',tool_calls=[tool('task_write',value='must-not-execute')])

def test_actual_five_response_shape_stops_at_four_without_spending_truncations():
    result,p,writes,spends=run_responses([missing(),truncated(),truncated(),truncated(),truncated()])
    assert result.stop_reason=='error' and result.error=='provider_truncation_exhausted: empty length responses=3'
    assert result.ix_contract_exhaustion is None and result.ix_contract_failure_kind is None
    assert len(p.calls)==4 and len(p.responses)==1 and writes==[]
    assert spends==[{'request':1,'counter':1}]
    assert [e['attempt'] for e in result.tool_events if e['name']=='provider_truncated']==[1,2,3]
    assert [e['attempts'] for e in result.tool_events if e['name']=='provider_truncation_exhausted']==[3]
    assert not any(e['name']=='ix_render_payload_forced' for e in result.tool_events)
    assert all(call[0]==p.calls[0][0] for call in p.calls)

def test_empty_provider_failure_without_ix_or_nudge_requests():
    result,p,writes,spends=run_responses([truncated()]*3,ix=False)
    assert result.stop_reason=='error' and result.ix_contract_exhaustion is None
    assert len(p.calls)==3 and not p.responses and writes==[] and spends==[]

def test_one_truncation_then_three_real_ix_misses_still_exhausts_legitimately():
    result,p,writes,spends=run_responses([truncated(),missing(),missing(),missing()])
    assert result.stop_reason=='ix_contract_failure'
    assert result.ix_contract_exhaustion=={'attempts':3,'contract':'render','operation':'required_render_carrier'}
    assert spends==[{'request':2,'counter':1},{'request':3,'counter':2},{'request':4,'counter':3}]
    assert len(p.calls)==4 and writes==[]

def test_successful_selection_survives_provider_retry():
    # Existing public selection schema from the prior regression suite.
    responses=[truncated(),truncated(),LLMResponse(content=None,tool_calls=[
        tool('IX_reasoning_visibility',setting='Show',evidence='Synthetic.',application='Give the reason.'),
        tool('IX_render_reasoning_visibility',user_facing_rationale='A synthetic reason.')]),LLMResponse(content='Completed.')]
    provider=ScriptedProvider(responses);base=ToolRegistry();context={'session_active_ix':['reasoning_visibility']}
    tools=TurnToolRegistry(base,['reasoning_visibility'],ix_context=context)
    result=asyncio.run(AgentRunner(provider).run(AgentRunSpec(initial_messages=[{'role':'user','content':'Original.'}],tools=tools,model='offline-test',required_tool_names=tools.required_ix_tool_names,ix_context=context,max_iterations=10)))
    assert result.stop_reason=='completed' and result.ix_contract_exhaustion is None
    assert len(provider.calls)==4 and not provider.responses
    assert any(e.get('name')=='IX_reasoning_visibility' and e.get('status')=='ok' for e in result.tool_events)

def test_provider_error_remains_separate_and_spends_no_ix():
    result,p,writes,spends=run_responses([LLMResponse(content='Synthetic provider error',finish_reason='error')]*2)
    assert result.stop_reason=='error' and result.ix_contract_exhaustion is None
    assert len(p.calls)==2 and spends==[] and writes==[]

def test_actual_tool_side_effect_is_retained_once_before_provider_failure():
    result,p,writes,spends=run_responses([LLMResponse(content=None,tool_calls=[tool('task_write',value='keep-once')]),truncated(),truncated(),truncated()],ix=False)
    assert result.stop_reason=='error' and writes==['keep-once'] and len(p.calls)==4
    assert result.ix_contract_exhaustion is None and spends==[]
    assert any(c.executed and c.tool_name=='task_write' for c in result.collector.calls)

def test_nonempty_length_response_remains_on_existing_native_path():
    result,p,writes,spends=run_responses([LLMResponse(content='A complete answer.',finish_reason='length')],ix=False)
    assert result.stop_reason=='completed' and result.final_content=='A complete answer.'
    assert len(p.calls)==1 and spends==[] and writes==[]
    assert not any(e['name']=='provider_truncation_exhausted' for e in result.tool_events)

@pytest.mark.asyncio
async def test_loop_provider_failure_keeps_events_without_persisting_fake_turn(tmp_path):
    from pa.agent.loop import AgentLoop,AgentProviderError
    from pa.bus.queue import MessageBus
    from test_memory_adapter_context import FakeAdapterForPA
    adapter=FakeAdapterForPA();provider=ScriptedProvider([truncated()]*3)
    loop=AgentLoop(bus=MessageBus(),pa_llm_provider=provider,workspace=tmp_path,pa_model='offline-test',memory_adapter=adapter)
    loop.sessions.create('test:memory')
    with pytest.raises(AgentProviderError,match='provider_truncation_exhausted') as error:
        await loop.process_direct_message('Must not persist.',session_key='test:memory',metadata={'run_id':'run_001','persona_id':'user_a'})
    assert len(provider.calls)==3
    assert loop.sessions.get('test:memory').messages==[] and adapter.record_events==[]
    assert len(error.value.failed_turn_events['iterations'])==3
    assert len([e for e in error.value.failed_turn_events['tool_events'] if e['name']=='provider_truncated'])==3
    assert not hasattr(error.value,'ix_contract_exhaustion')
    from harness.orchestrator import _three_miss_ix_exhaustion
    assert _three_miss_ix_exhaustion(error.value) is None
