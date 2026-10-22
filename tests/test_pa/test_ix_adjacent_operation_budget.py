"""Real carrier success and a new confirmation gate may share one response."""
import asyncio
import socket
import pytest
from test_runner_ix import ScriptedProvider, _ix_call, _plan, _task, _NamedEchoTool
from pa.agent.runner import AgentRunner, AgentRunSpec
from pa.agent.ix import TurnToolRegistry
from pa.agent.tools.registry import ToolRegistry
from pa.providers.base import LLMResponse

@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setattr(socket.socket,'connect',lambda *a,**k:(_ for _ in ()).throw(AssertionError('offline test')))

def case(misses=1, *, final_ask=False, unrelated=False, invalid_selection=False, invalid_render=False, invalid_binding=False):
    writes=[]
    class Write(_NamedEchoTool):
        async def execute(self,value):writes.append(value);return {'ok':True}
    selection=_ix_call('IX_autonomy_level',setting='invalid' if invalid_selection else 'Self-directed')
    render=_plan({'id':'note','bound_item_ids':['note1']},{'id':'send','bound_item_ids':['send1']})
    if invalid_render:render.arguments={'plan_id':'missing_plan'}
    if invalid_binding:render=_plan({'id':'wrong','bound_item_ids':['not-the-business-item']})
    combined=LLMResponse(content=None,tool_calls=[selection,render,_task('note','task_note','note1'),_task('send1','task_send','send1')])
    if unrelated:
        combined.tool_calls.insert(0,_ix_call('IX_render_reasoning_visibility',user_facing_rationale='Initial reason.'))
    responses=[*[LLMResponse(content='Missing carrier.') for _ in range(misses)],combined]
    if invalid_selection or invalid_render or invalid_binding:
        responses.extend([combined,combined,LLMResponse(content='no fourth')])
    else:
        for n in (2,3):
            tools=[_task(f'send{n}','task_send','send1',ask='Send now?' if final_ask and n==3 else '')]
            if unrelated:
                # Already banked selection is deduplicated; unrelated successful
                # render must not turn the same unresolved gate into a new one.
                tools.insert(0,_ix_call('IX_autonomy_level',setting='Autonomous'))
                tools.insert(0,_ix_call('IX_render_reasoning_visibility',user_facing_rationale='Reason.'))
            responses.append(LLMResponse(content=None,tool_calls=tools))
        responses.append(LLMResponse(content='Must not request fourth gate attempt.'))
    provider=ScriptedProvider(responses);base=ToolRegistry();base.register(Write('task_note'));base.register(Write('task_send'))
    context={'session_active_ix':['autonomy_level','reasoning_visibility'] if unrelated else ['autonomy_level'],
             'tool_action_types':{'task_note':'internal_write','task_send':'external_action'}}
    tools=TurnToolRegistry(base,['autonomy_level'],ix_context=context)
    result=asyncio.run(AgentRunner(provider).run(AgentRunSpec(initial_messages=[{'role':'user','content':'same ACC and context'}],tools=tools,model='offline',max_iterations=12,required_tool_names=tools.required_ix_tool_names,ix_context=context)))
    return result,provider,writes

@pytest.mark.parametrize('misses',[1,2])
@pytest.mark.parametrize('final_ask',[False,True])
def test_fresh_gate_has_three_attempts_after_actual_carrier_success(misses,final_ask):
    result,provider,writes=case(misses,final_ask=final_ask)
    assert len(provider.calls)==misses+3
    assert writes==['note1']
    assert all(m[0]==provider.calls[0][0] for m in provider.calls)
    assert len(provider.responses)==1
    assert result.stop_reason==('ix_gate' if final_ask else 'ix_contract_failure')
    if final_ask:assert result.ix_contract_exhaustion is None and result.final_content=='Send now?'
    else:assert result.ix_contract_exhaustion=={'attempts':3,'contract':'gate','operation':'gate_confirmation_message'}
    assert sum(e.get('name')=='ix_gate_no_message' for e in result.tool_events)==(2 if final_ask else 3)

@pytest.mark.parametrize('fault',['invalid_selection','invalid_render'])
def test_received_but_invalid_carrier_does_not_reset_pending_operation(fault):
    result,provider,writes=case(1,**{fault:True})
    assert result.stop_reason=='ix_contract_failure'
    assert result.ix_contract_exhaustion['attempts']==3
    assert len(provider.calls)==3 and not writes
    assert not any(e.get('name')=='ix_gate_no_message' for e in result.tool_events)


def test_repeated_changed_selection_and_unrelated_ix_do_not_reset_gate():
    result,provider,writes=case(1,unrelated=True)
    assert result.stop_reason=='ix_contract_failure'
    assert result.ix_contract_exhaustion=={'attempts':3,'contract':'gate','operation':'gate_confirmation_message'}
    assert len(provider.calls)==4 and writes==['note1']
    assert sum(e.get('name')=='IX_autonomy_level' and e.get('ix') for e in result.tool_events)==1


def test_schema_valid_but_unbound_payload_stays_ordinary_binding_failure():
    result,provider,writes=case(1,invalid_binding=True)
    assert not writes
    assert result.ix_contract_exhaustion is None
    assert not any(e.get('name')=='ix_gate_no_message' for e in result.tool_events)
    assert any(e.get('name')=='ix_binding_invalid' for e in result.tool_events)
    assert len(provider.calls)<=4
