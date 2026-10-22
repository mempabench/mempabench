"""Frozen D066: a no-write failed attempt is not a second outward act."""
import asyncio
import hashlib
import json
import shutil
from pathlib import Path

import pytest
import yaml

from harness.capability_attempt import CONTEXT_KEY, current, prepare
from harness.gate import resolve_authorised_tool
from pa.agent.ix.collector import TurnCollector
from pa.agent.ix.events import IXDirective
from pa.agent.ix.render_types import ConsentDecision
from pa.agent.runner import AgentRunSpec, AgentRunner
from pa.providers.base import ToolCallRequest
from state_server.store import StateStore

ROOT = Path(__file__).resolve().parents[2]
DRAFT = ROOT / 'Authoring/sessions/normal/drafts/user_d/acc_066.yaml'
ARGS = dict(product_id='shp_np_letterhead', quantity=1,
            delivery_address_id='addr_2c8hrw', payment_method_id='pm_branch_7702')


@pytest.fixture
def world(tmp_path):
    state = tmp_path / 'state'
    shutil.copytree(ROOT / 'data/state_fixtures/user_d', state)
    store = StateStore(state, run_id='test', user_id='user_d', session_id='acc_066',
                       session_timestamp='2026-11-20T15:00:00')
    permits = prepare(DRAFT, state, session_timestamp='2026-11-20T15:00:00')
    return state, store, permits


def hashes(state):
    return {str(p.relative_to(state)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in state.rglob('*') if p.is_file() and p.name != 'tool_log.jsonl'}


class Tools:
    tool_names = ['shopping_order_create', 'bill_payment_create', 'document_update']
    def __init__(self, store):
        self.store = store
        self.calls = []
    async def execute(self, name, args):
        self.calls.append((name, args))
        method = 'update_document' if name == 'document_update' else name
        return json.dumps(getattr(self.store, method)(**args))


def execute(store, permits, calls, consent=True):
    tools = Tools(store)
    spec = AgentRunSpec(initial_messages=[], tools=tools, model='unused', max_iterations=1,
        ix_context={CONTEXT_KEY: permits},
        consent_decisions=[ConsentDecision(interaction_item_id='', decision='allow',
                                          tool='bill_payment_create')] if consent else [])
    collector = TurnCollector()
    runner = AgentRunner(None)
    result = asyncio.run(runner._execute_tools_ix_first(
        spec, calls, IXDirective(autonomy_level_policy='confirm_key_actions'), collector))
    return tools, collector, result


def call(name='shopping_order_create', args=None, id='shop'):
    return ToolCallRequest(id=id, name=name, arguments=ARGS.copy() if args is None else args)


def test_frozen066_real_limit_then_only_88_payment(world):
    state, store, permits = world
    assert len(permits) == 1
    assert resolve_authorised_tool(DRAFT, ROOT, capability_limit_attempts=permits) == 'bill_payment_create'
    before = hashes(state)
    tools, collector, (_, events) = execute(store, permits, [call()])
    assert len(tools.calls) == 1
    assert events[-1]['status'] == 'capability_limit'
    assert collector.capability_limit_seen
    assert hashes(state) == before
    # The existing standing grant, unchanged, permits the actual recovery.
    payment = dict(bill_id='GQP-2026-1120', payment_method_id='pm_branch_7702', amount=88.0)
    tools, collector, (_, events) = execute(store, permits, [call('bill_payment_create', payment, 'pay')])
    assert len(tools.calls) == 1 and events[-1]['status'] == 'ok'
    bill = next(b for b in json.loads((state/'finance/bills.json').read_text())['bills']
                if b['bill_id'] == 'GQP-2026-1120')
    assert bill['status'] == 'paid'
    assert tools.calls[0][1]['amount'] == 88.0


@pytest.mark.parametrize('change', [dict(quantity=2), dict(product_id='other'),
                                   dict(payment_method_id='other'), dict(extra='x'), dict(quantity=True), dict(quantity=1.0)])
def test_wrong_arguments_still_hit_floor(world, change):
    state, store, permits = world
    before = hashes(state)
    tools, collector, _ = execute(store, permits, [call(args=ARGS | change)])
    assert tools.calls == [] and collector.gate.reason == 'external_action_floor'
    assert hashes(state) == before


def test_earlier_turn_has_no_permit(world):
    state, store, _ = world
    tools, collector, _ = execute(store, [], [call()], consent=False)
    assert tools.calls == [] and collector.gate.reason == 'external_action_floor'


def test_current_world_change_refuses_resolver_and_execution(world):
    state, store, permits = world
    path = state/'shopping/products.json'
    data = json.loads(path.read_text())
    next(p for p in data['products'] if p['product_id'] == ARGS['product_id'])['unavailable'] = False
    path.write_text(json.dumps(data))
    assert not current(permits[0])
    with pytest.raises(ValueError, match='2 different tools'):
        resolve_authorised_tool(DRAFT, ROOT, capability_limit_attempts=permits)
    tools, collector, _ = execute(store, permits, [call()])
    assert not tools.calls and collector.gate.reason == 'external_action_floor'


def test_same_batch_mutation_is_rechecked_at_dispatch(world):
    state, store, permits = world
    path = state/'shopping/products.json'
    data = json.loads(path.read_text())
    next(p for p in data['products'] if p['product_id'] == ARGS['product_id'])['unavailable'] = False
    mutation = call('document_update', dict(path='shopping/products.json', anchor=path.read_text(), replacement=json.dumps(data)), 'change')
    tools, collector, _ = execute(store, permits, [mutation, call(), call('bill_payment_create',
        dict(bill_id='GQP-2026-1120', payment_method_id='pm_branch_7702', amount=88.0), 'pay')])
    assert [n for n, _ in tools.calls] == ['document_update']
    assert collector.record_for('shop').executed is False
    assert collector.record_for('pay').executed is False
    assert collector.gate.reason == 'external_action_floor'


def test_flag_and_dynamic_refs_do_not_exempt_two_outward_tools(world, tmp_path):
    _, _, permits = world
    draft = yaml.safe_load(DRAFT.read_text())
    step = next(s for s in draft['task_execution']['atomic_steps'] if s['step_id'] == 'try_order')
    step['tool_arguments']['product_id'] = '$find_letterhead.products.0.product_id'
    path = tmp_path/'draft.yaml'
    path.write_text(yaml.safe_dump(draft))
    with pytest.raises(ValueError, match='2 different tools'):
        resolve_authorised_tool(path, ROOT, capability_limit_attempts=permits)
    with pytest.raises(ValueError, match='2 different tools'):
        resolve_authorised_tool(DRAFT, ROOT)


def test_passive_authorisation_uses_real066_beat_not_earlier_or_reaction():
    from harness.run_plan_builder import _load_script
    from simulator.loop import _preauthorised_tool
    from simulator.message_quality import beat_carries_authorisation
    script = _load_script(DRAFT).model_copy(update={'authorised_tool': 'bill_payment_create'})
    message = 'I authorize you to try the order. If blocked, pay the existing invoice within $100.'
    assert _preauthorised_tool(script, script.beats[0], message, []) == 'bill_payment_create'
    assert _preauthorised_tool(script, script.beats[1], message, []) is None
    assert _preauthorised_tool(script, script.beats[0],
        'You may pay the invoice if the supplier blocks the order.', []) is None
    assert _preauthorised_tool(script, script.beats[0], 'Just check the price for now.', []) is None
    assert _preauthorised_tool(script, script.beats[0], message, [{'id': 'held'}]) is None
    for text in ['Paying the invoice is not permitted.',
                 'Paying is permitted only after confirmation.',
                 'Paying is permitted after user confirmation.']:
        assert not beat_carries_authorisation(text)


def test_frozen066_via_real_mcp_and_ix_runner(world):
    from contextlib import AsyncExitStack
    from harness.state_runtime import StageStepRuntime
    from pa.agent.tools.mcp import connect_mcp_servers
    from pa.agent.tools.registry import ToolRegistry
    state, _, permits = world
    async def scenario():
        runtime = StageStepRuntime(run_id='test', user_id='user_d', session_id='acc_066',
            step_id='acc_066', state_root=state, project_root=ROOT,
            enabled_tools=['shopping_order_create', 'bill_payment_create'],
            session_timestamp='2026-11-20T15:00:00')
        async with AsyncExitStack() as stack:
            tools = ToolRegistry()
            await connect_mcp_servers(runtime.mcp_server_config(), tools, stack)
            spec = AgentRunSpec(initial_messages=[], tools=tools, model='unused', max_iterations=1,
                ix_context={CONTEXT_KEY: permits},
                consent_decisions=[ConsentDecision('', 'allow', tool='bill_payment_create')])
            runner, collector = AgentRunner(None), TurnCollector()
            before = hashes(state)
            _, events = await runner._execute_tools_ix_first(spec,
                [call('mcp_state_shopping_order_create')],
                IXDirective(autonomy_level_policy='confirm_key_actions'), collector)
            assert events[-1]['status'] == 'capability_limit'
            assert hashes(state) == before
            assert collector.capability_limit_seen
            _, events = await runner._execute_tools_ix_first(spec,
                [call('mcp_state_bill_payment_create', dict(bill_id='GQP-2026-1120',
                    payment_method_id='pm_branch_7702', amount=88.0), 'pay')],
                IXDirective(autonomy_level_policy='confirm_key_actions'), collector)
            assert events[-1]['status'] == 'ok'
            assert [r.status for r in collector.calls] == ['capability_limit', 'ok']
    asyncio.run(scenario())


def test_ix_carriers_are_not_business_arguments(world):
    _, store, permits = world
    tools, collector, (_, events) = execute(store, permits, [call(args=ARGS | {
        'interaction_item_id': 'try_order', '_ix_confirmation_message': 'Try the order?'
    })])
    assert tools.calls == [('shopping_order_create', ARGS)]
    assert events[-1]['status'] == 'capability_limit'
    assert collector.calls[0].interaction_item_id == 'try_order'


@pytest.mark.parametrize('number,clock,recovery', [
    (101, '2027-01-16T12:00:00', 'email_message_send'),
    (102, '2027-01-19T19:00:00', 'appointment_create'),
])
def test_other_frozen_limits_use_real_mcp_without_mutation(world, number, clock, recovery):
    from contextlib import AsyncExitStack
    from harness.state_runtime import StageStepRuntime
    from harness.run_plan_builder import _load_script
    from simulator.loop import _preauthorised_tool
    from pa.agent.tools.mcp import connect_mcp_servers
    from pa.agent.tools.registry import ToolRegistry
    state, _, _ = world
    draft = DRAFT.with_name(f'acc_{number}.yaml')
    before = hashes(state)
    attempts = prepare(draft, state, session_timestamp=clock)
    assert len(attempts) == 1 and hashes(state) == before  # verifier does not execute handlers
    assert resolve_authorised_tool(draft, ROOT, capability_limit_attempts=attempts) == recovery
    script = _load_script(draft).model_copy(update={'authorised_tool': recovery})
    assert _preauthorised_tool(script, script.beats[0], 'I authorize you to try the requested change.', []) == recovery
    assert _preauthorised_tool(script, script.beats[0], 'Only if the attempt fails, you may book it.', []) is None
    assert _preauthorised_tool(script, script.beats[-1], 'I authorize you to proceed.', []) is None
    attempt = attempts[0]
    async def scenario():
        runtime = StageStepRuntime(run_id='test', user_id='user_d', session_id=f'acc_{number}',
            step_id=f'acc_{number}', state_root=state, project_root=ROOT,
            enabled_tools=[attempt['tool'], recovery], session_timestamp=clock)
        async with AsyncExitStack() as stack:
            tools = ToolRegistry()
            await connect_mcp_servers(runtime.mcp_server_config(), tools, stack)
            spec = AgentRunSpec(initial_messages=[], tools=tools, model='unused', max_iterations=1,
                ix_context={CONTEXT_KEY: attempts},
                consent_decisions=[ConsentDecision('', 'allow', tool=recovery)])
            runner, collector = AgentRunner(None), TurnCollector()
            wrong = attempt['arguments'] | ({'file_id': 'documents/other.csv'} if number == 101
                                           else {'remove_holder': 'Samuel Wexler'})
            refused = TurnCollector()
            await runner._execute_tools_ix_first(spec,
                [call('mcp_state_' + attempt['tool'], wrong)],
                IXDirective(autonomy_level_policy='confirm_key_actions'), refused)
            assert refused.gate.reason == 'external_action_floor' and hashes(state) == before
            from dataclasses import replace
            early = TurnCollector()
            await runner._execute_tools_ix_first(replace(spec, ix_context={}),
                [call('mcp_state_' + attempt['tool'], attempt['arguments'])],
                IXDirective(autonomy_level_policy='confirm_key_actions'), early)
            assert early.gate.reason == 'external_action_floor' and hashes(state) == before
            _, events = await runner._execute_tools_ix_first(spec,
                [call('mcp_state_' + attempt['tool'], attempt['arguments'])],
                IXDirective(autonomy_level_policy='confirm_key_actions'), collector)
            assert events[-1]['status'] == 'capability_limit' and hashes(state) == before
            raw = yaml.safe_load(draft.read_text())
            step = next(s for s in raw['task_execution']['atomic_steps'] if s.get('tool') == recovery)
            args = step['tool_arguments'].copy()
            if number == 101:
                contacts = json.loads((state/'contacts.json').read_text())['contacts']
                args['to'] = next(c['email'] for c in contacts if c['name'] == 'Cascade corporate service desk')
                args['attachments'] = [attempt['arguments']['file_id']]
            else:
                providers = json.loads((state/'appointment/availability.json').read_text())['providers']
                args['provider_id'] = next(p['provider_id'] for p in providers
                    if 'identity verification' in json.dumps(p).lower())
            _, events = await runner._execute_tools_ix_first(spec,
                [call('mcp_state_' + recovery, args, 'recovery')],
                IXDirective(autonomy_level_policy='confirm_key_actions'), collector)
            assert events[-1]['status'] == 'ok'
    asyncio.run(scenario())


@pytest.mark.parametrize('case', ['search_projection', 'wrong_projection', 'missing_dependency',
                                 'intervening_write', 'directory', 'escape', 'noncanonical',
                                 'future', 'changed_report', 'unlocked', 'wrong_holder'])
def test_other_limits_fail_closed(world, tmp_path, case):
    state, _, _ = world
    number = 102 if case == 'wrong_holder' else 101
    path = DRAFT.with_name(f'acc_{number}.yaml')
    raw = yaml.safe_load(path.read_text())
    steps = raw['task_execution']['atomic_steps']
    if number == 101:
        source = next(s for s in steps if s['step_id'] == 'read_report')
        target = next(s for s in steps if s['step_id'] == 'try_submit')
        if case == 'search_projection': source['tool'] = 'document_search'
        if case == 'wrong_projection': target['tool_arguments']['file_id'] = '$read_report.files.0'
        if case == 'missing_dependency': target['depends_on'] = []
        if case == 'intervening_write':
            steps.insert(steps.index(target), dict(step_id='rewrite', actor='tool', tool='document_update', action_kind='write'))
        if case == 'directory': source['tool_arguments']['path'] = 'documents/branch'
        if case == 'escape': source['tool_arguments']['path'] = '../secret'
        if case == 'noncanonical': source['tool_arguments']['path'] = 'documents/branch/../branch/monthly_branch_report_2027-01.csv'
    edited = tmp_path/'draft.yaml'
    edited.write_text(yaml.safe_dump(raw))
    clock = '2027-01-15T12:00:00' if case == 'future' else '2027-01-19T19:00:00'
    attempts = prepare(edited, state, session_timestamp=clock)
    if case in {'changed_report', 'unlocked', 'wrong_holder'}:
        assert len(attempts) == 1
        if case == 'changed_report':
            (state / attempts[0]['arguments']['file_id']).write_text('changed report')
        elif case == 'unlocked':
            path = Path(attempts[0]['backing_path'])
            data = json.loads(path.read_text())
            data['accounts'][0]['unavailable'] = False
            path.write_text(json.dumps(data))
        else:
            from harness.capability_attempt import matches
            assert not matches(attempts[0], attempts[0]['tool'], attempts[0]['arguments'] | {'remove_holder': 'Samuel Wexler'})
            return
        assert not current(attempts[0])
    else:
        assert attempts == []
    with pytest.raises(ValueError, match='2 different tools'):
        resolve_authorised_tool(edited, ROOT, capability_limit_attempts=attempts)


@pytest.mark.parametrize('text', [
    'Does not permit a service-desk unlock email.',
    'Permits an identity-check booking only after confirmation.',
    '"Permits a service-desk unlock email" is wording she is asking about.',
])
def test_active_permit_detection_does_not_treat_denial_or_quotes_as_authorisation(text):
    from simulator.message_quality import beat_carries_authorisation
    assert not beat_carries_authorisation(text)


def test_runtime_descriptor_never_enters_pa_prompt(world, tmp_path):
    from .conftest import ScriptProvider, build_loop, run_turn
    from pa.providers.base import LLMResponse
    _, _, permits = world
    provider = ScriptProvider([LLMResponse(content='I will check the available route.')])
    loop, _ = build_loop(tmp_path/'pa', provider)
    asyncio.run(run_turn(loop, 'Check the available route.', active_ix=[],
                         extra_ix_context={CONTEXT_KEY: permits}))
    prompt = json.dumps(provider.requests, default=str)
    assert CONTEXT_KEY not in prompt
    assert permits[0]['backing_sha256'] not in prompt
    assert permits[0]['backing_path'] not in prompt
