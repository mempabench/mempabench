"""Real installed Mem0 parsing/persistence with offline storage and provider stubs."""
import json
import socket
import stat
import traceback
from types import SimpleNamespace

import pytest

from memory.backends.mem0_client import _ScopedMem0Client, _MemoryExtractionLLM
from pa.providers.base import LLMResponse

def llm_for(tmp_path, monkeypatch, responses):
    llm = object.__new__(_MemoryExtractionLLM)
    llm._failure_evidence_dir = tmp_path / 'private_memory_diagnostics'
    llm._logical_task_key = None
    llm._model = 'deepseek/deepseek-v4-flash'
    llm._request_succeeded = False
    llm.extraction_max_output_tokens = 32000
    calls, remaining = [], iter(responses)
    def call(**kwargs):
        calls.append(kwargs)
        return next(remaining)
    llm._call_provider = call
    monkeypatch.setattr('memory.backends.mem0_client.time.sleep', lambda _: None)
    return llm, calls


def evidence(llm):
    return [json.loads(p.read_text()) for p in llm._failure_evidence_dir.glob('*.json')]



@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setattr(socket.socket, 'connect', lambda *a, **k: (_ for _ in ()).throw(AssertionError('network forbidden')))


def native_client(tmp_path, monkeypatch, payload, *, finish='length', fail=None):
    from mem0 import Memory
    import mem0.memory.main as native
    monkeypatch.setattr(native, 'lemmatize_for_bm25', lambda text: text)
    monkeypatch.setattr(native, 'extract_entities_batch', lambda texts: [[] for _ in texts])
    monkeypatch.setattr(native, 'capture_event', lambda *a, **k: None)
    llm, calls = llm_for(tmp_path, monkeypatch, [LLMResponse(payload, finish_reason=finish)] * (3 if finish=='stop' else 1))
    memory = object.__new__(Memory)
    memory.custom_instructions = None
    memory.api_version = 'v1.1'
    memory.config = SimpleNamespace(llm=SimpleNamespace(config={'enable_vision':False}))
    points, history, saved = {}, [], []
    def insert(**kwargs):
        if fail=='vectors':raise ValueError('PRIVATE_NATIVE_FAILURE ' + str(payload))
        points.update({k:v for k,v in zip(kwargs['ids'],kwargs['payloads'])})
    def save_messages(*args, **kwargs):
        if fail=='messages':raise ValueError('PRIVATE_NATIVE_FAILURE ' + str(payload))
        saved.append((args,kwargs))
    def add_history(records):
        if fail=='history':raise RuntimeError('PRIVATE_NATIVE_FAILURE ' + str(payload))
        history.extend(records)
    memory.db = SimpleNamespace(get_last_messages=lambda *a, **k: [], save_messages=save_messages, batch_add_history=add_history)
    memory.embedding_model = SimpleNamespace(embed=lambda *a, **k: [.1], embed_batch=lambda texts,*a,**k:[[.1] for _ in texts])
    memory.vector_store = SimpleNamespace(search=lambda **k:[], insert=insert,
        get=lambda vector_id:SimpleNamespace(id=vector_id,payload=points[vector_id]) if vector_id in points else None)
    memory.llm = llm
    client = _ScopedMem0Client(memory,llm,collection_name='owned',qdrant_host='127.0.0.1',qdrant_port=6333,
        expected_run_id='offline',embedding_dimensions=1,qdrant_distance='Cosine',qdrant_on_disk=True,
        qdrant_bm25_enabled=False,enable_entity_memory=False)
    scope={'run_id':'offline','user_id':'user_a','metadata':{'session_id':'acc_011','turn_id':'beat1:t1','timestamp':'2026-09-21T00:00:00+00:00'}}
    def add():return client.add([{'role':'user','content':'offline turn'}],**scope)
    return add,llm,calls,points,history,saved


@pytest.mark.parametrize('payload', [None, '', ' ', '{}', '{"wrong":[]}', '{"memory":null}', '{"memory":[]}', '{"memory":[{}]}'])
def test_native_no_change_is_accepted_without_new_facts(tmp_path,monkeypatch,payload):
    add,llm,calls,points,history,saved=native_client(tmp_path,monkeypatch,payload)
    assert add()=={'results':[]}
    assert len(calls)==1 and llm.request_succeeded and len(saved)==1 and not points and not history
    rows=evidence(llm);assert len(rows)==1 and rows[0]['reason']=='length_observed'
    assert rows[0]['response']['content']==payload
    assert rows[0]['native_forwarding']=={'content':payload or '', 'none_normalized_to_empty_string':payload is None}
    assert stat.S_IMODE(llm._failure_evidence_dir.stat().st_mode)==0o700
    assert all(stat.S_IMODE(p.stat().st_mode)==0o600 for p in llm._failure_evidence_dir.iterdir())


@pytest.mark.parametrize('payload', ['{"memory":[{"text":"real fact"}]}',
    '```json\n{"memory":[{"text":"real fact"}]}\n```',
    'prefix {"memory":[{"text":"real fact"}]} trailing text'])
def test_native_valid_length_persists_original_fact(tmp_path,monkeypatch,payload):
    add,llm,calls,points,history,saved=native_client(tmp_path,monkeypatch,payload)
    result=add();assert len(result['results'])==len(points)==len(history)==len(saved)==1
    assert result['results'][0]['memory']=='real fact' and len(calls)==1
    assert evidence(llm)[0]['response']['content']==payload


@pytest.mark.parametrize('payload', ['{"memory":[{"text":"PRIVATE_RAW_MARKER"},', 'PRIVATE_RAW_MARKER{', '["PRIVATE_RAW_MARKER"]'])
def test_native_parse_error_is_real_private_and_not_retried(tmp_path,monkeypatch,payload,caplog):
    add,llm,calls,points,history,saved=native_client(tmp_path,monkeypatch,payload)
    try:add()
    except RuntimeError as exc:
        formatted=traceback.format_exc()
        assert exc.__suppress_context__ and 'native_error_type=ValueError' in str(exc)
    else:pytest.fail('genuine native parse error was swallowed')
    assert 'PRIVATE_RAW_MARKER' not in formatted+caplog.text
    assert len(calls)==1 and not llm.request_succeeded and not points and not history and not saved
    rows=evidence(llm);assert {r['reason'] for r in rows}=={'length_observed','native_memory_failure_after_length'}
    failed=next(r for r in rows if r['validation_error'])
    assert payload in failed['validation_error']['message'] or 'PRIVATE_RAW_MARKER' in failed['validation_error']['message']
    assert all(r['response']['content']==payload for r in rows)


@pytest.mark.parametrize('fail',['vectors','history','messages'])
def test_native_storage_failure_preserves_partial_effects_and_private_cause(tmp_path,monkeypatch,fail,caplog):
    payload='{"memory":[{"text":"PRIVATE_RAW_MARKER"}]}'
    add,llm,calls,points,history,saved=native_client(tmp_path,monkeypatch,payload,fail=fail)
    try:add()
    except RuntimeError:formatted=traceback.format_exc()
    else:pytest.fail('storage failure swallowed')
    assert len(calls)==1 and not llm.request_succeeded
    assert 'PRIVATE_RAW_MARKER' not in formatted+caplog.text and 'PRIVATE_NATIVE_FAILURE' not in formatted+caplog.text
    assert bool(points)==(fail!='vectors')
    assert bool(history)==(fail=='messages')
    assert not saved
    assert any('PRIVATE_NATIVE_FAILURE' in (r['validation_error'] or {}).get('message','') for r in evidence(llm))


def test_stop_schema_still_retries_three_times(tmp_path,monkeypatch):
    add,llm,calls,points,history,saved=native_client(tmp_path,monkeypatch,'{"wrong":[]}',finish='stop')
    with pytest.raises(RuntimeError,match='extraction failed'):add()
    assert len(calls)==3 and not llm.request_succeeded and not points and not history and not saved
    assert {r['reason'] for r in evidence(llm)}=={'json_contract'}


def test_next_task_does_not_inherit_old_length_exception_redaction(tmp_path,monkeypatch):
    add,llm,calls,points,history,saved=native_client(tmp_path,monkeypatch,'')
    add();assert llm._length_response_evidence is not None
    llm.set_logical_task_key('next')
    assert llm._length_response_evidence is None and not llm.request_succeeded
