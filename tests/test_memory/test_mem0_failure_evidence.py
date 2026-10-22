"""No-network failure injection for private extraction evidence."""
import json
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest

from memory.backends.mem0_client import _MemoryExtractionLLM
from pa.providers.base import LLMResponse


def llm_for(tmp_path, monkeypatch, responses):
    llm = object.__new__(_MemoryExtractionLLM)
    llm._failure_evidence_dir = tmp_path / 'private_memory_diagnostics'
    llm._logical_task_key = 'mem0_add:run:session:acc_010:turn5'
    llm._model = 'deepseek/deepseek-v4-flash'
    llm._request_succeeded = False
    llm.extraction_max_output_tokens = 32000
    llm.api_key = 'secret-not-to-be-written'
    calls = []
    it = iter(responses)
    def call(**kwargs):
        calls.append(kwargs)
        return next(it)
    llm._call_provider = call
    monkeypatch.setattr('memory.backends.mem0_client.time.sleep', lambda _: None)
    return llm, calls


def invoke(llm):
    return llm.generate_response(messages=[{'role': 'user', 'content': 'exact input'}],
                                 response_format={'type': 'json_object'})


def evidence(llm):
    return [json.loads(p.read_text()) for p in llm._failure_evidence_dir.glob('*.json')]


def test_length_full_content_one_call_private_no_credentials(tmp_path, monkeypatch):
    content = '{"memory":[' + 'x' * 101130
    llm, calls = llm_for(tmp_path, monkeypatch, [LLMResponse(content, finish_reason='length', usage={'completion_tokens': 32000})])
    assert invoke(llm) == content
    saved = [x for x in evidence(llm) if x["reason"] == "length_observed"]
    assert len(calls) == len(saved) == 1
    assert saved[0]['response']['content'] == content
    assert saved[0]['response']['usage'] == {'completion_tokens': 32000}
    assert saved[0]['request']['messages'] == calls[0]['messages']
    assert saved[0]['request']['max_tokens'] == 32000
    assert 'secret-not-to-be-written' not in json.dumps(saved)
    assert stat.S_IMODE(llm._failure_evidence_dir.stat().st_mode) == 0o700
    assert all(stat.S_IMODE(p.stat().st_mode) == 0o600 for p in llm._failure_evidence_dir.iterdir())
    assert llm.request_succeeded


def test_three_json_failures_keep_separate_exact_causes(tmp_path, monkeypatch):
    replies = ['{', '{"wrong":[]}', '{"memory":[{"text":""}]}']
    llm, calls = llm_for(tmp_path, monkeypatch, [LLMResponse(x) for x in replies])
    with pytest.raises(ValueError, match='attempt=3/3'):
        invoke(llm)
    saved = sorted(evidence(llm), key=lambda x: x['contract_attempt'])
    assert len(calls) == 3
    assert [x['response']['content'] for x in saved] == replies
    assert [x['validation_error']['type'] for x in saved] == ['JSONDecodeError', 'ValueError', 'ValueError']
    assert 'memory list' in saved[1]['validation_error']['message']
    assert 'non-empty text' in saved[2]['validation_error']['message']
    assert all(call == calls[0] for call in calls)


def test_success_no_evidence(tmp_path, monkeypatch):
    llm, calls = llm_for(tmp_path, monkeypatch, [LLMResponse('{"memory":[]}')])
    assert invoke(llm) == '{"memory":[]}'
    assert llm.request_succeeded and len(calls) == 1
    assert not llm._failure_evidence_dir.exists()


def test_io_failure_preserves_original_contract_and_no_payload_log(tmp_path, monkeypatch, caplog):
    llm, calls = llm_for(tmp_path, monkeypatch, [LLMResponse('private body')] * 3)
    llm._failure_evidence_dir.mkdir(mode=0o755)
    with pytest.raises(ValueError, match='attempt=3/3'):
        invoke(llm)
    assert len(calls) == 3 and not list(llm._failure_evidence_dir.iterdir())
    assert 'PermissionError' in caplog.text
    assert 'private body' not in caplog.text


def test_symlink_directory_rejected(tmp_path, monkeypatch):
    llm, calls = llm_for(tmp_path, monkeypatch, [LLMResponse('x', finish_reason='length')])
    target = tmp_path / 'other'; target.mkdir(mode=0o700)
    llm._failure_evidence_dir.symlink_to(target)
    assert invoke(llm) == 'x'
    assert len(calls) == 1 and not list(target.iterdir())


@pytest.mark.parametrize('collision', ['final', 'pending'])
def test_collision_never_overwrites_or_deletes_old_evidence(tmp_path, monkeypatch, collision):
    llm, calls = llm_for(tmp_path, monkeypatch, [LLMResponse('x', finish_reason='length')])
    llm._failure_evidence_dir.mkdir(mode=0o700)
    monkeypatch.setattr('memory.backends.mem0_client.uuid.uuid4', lambda: SimpleNamespace(hex='same'))
    old = llm._failure_evidence_dir / ('same.json' if collision == 'final' else '.same.json.pending')
    old.write_bytes(b'old immutable evidence')
    assert invoke(llm) == 'x'
    assert len(calls) == 1 and old.read_bytes() == b'old immutable evidence'
    assert list(llm._failure_evidence_dir.iterdir()) == [old]


def test_mainline_branch_evidence_separate(tmp_path, monkeypatch):
    for root in [tmp_path / 'run', tmp_path / 'run/probe_branches/p01']:
        root.mkdir(parents=True, exist_ok=True)
        llm, _ = llm_for(root, monkeypatch, [LLMResponse('x', finish_reason='length')])
        assert invoke(llm) == 'x'
    assert len(list(tmp_path.rglob('*.json'))) == 2


def test_length_valid_payload_accepted_with_length_evidence(tmp_path, monkeypatch):
    payload = '{"memory":[{"text":"a real memory"}]}'
    llm, calls = llm_for(tmp_path, monkeypatch, [LLMResponse(payload, finish_reason='length')])
    assert invoke(llm) == payload
    assert llm.request_succeeded and len(calls) == 1
    assert evidence(llm)[0]['response']['finish_reason'] == 'length'
    assert evidence(llm)[0]['reason'] == 'length_observed'
    assert evidence(llm)[0]['run_status'] == 'not_evaluated'
    assert evidence(llm)[0]['response']['content'] == payload


@pytest.mark.parametrize('payload', ['', '{"other":[]}', '{"memory":[{}]}', '{"memory":null}'])
def test_length_does_not_prevalidate_or_repair_or_retry(tmp_path, monkeypatch, payload):
    llm, calls = llm_for(tmp_path, monkeypatch, [LLMResponse(payload, finish_reason='length')])
    assert invoke(llm) == payload
    assert len(calls) == 1 and llm.request_succeeded


def test_provider_error_semantics_unchanged(tmp_path, monkeypatch):
    llm, calls = llm_for(tmp_path, monkeypatch, [LLMResponse('upstream failed', finish_reason='error')])
    with pytest.raises(RuntimeError, match='request failed'):
        invoke(llm)
    assert len(calls) == 1 and not llm.request_succeeded
    assert evidence(llm)[0]['reason'] == 'error'


@pytest.mark.parametrize('payload', [
    '```json\n{"memory":[]}\n```',
    '<think>reasoning</think>{"memory":[{"text":"x"}]}',
    'Here is the result: {"memory":[{"text":"x"}]} trailing prose',
    '```json\n{"memory":[{"text":"x"}]}\n``` unfinished prose',
])
def test_length_accepts_native_readable_content_without_rewriting(tmp_path, monkeypatch, payload):
    llm, calls = llm_for(tmp_path, monkeypatch, [LLMResponse(payload, finish_reason='length')])
    assert invoke(llm) == payload
    assert llm.request_succeeded and len(calls) == 1
    assert evidence(llm)[0]['response']['content'] == payload


def test_stop_fence_still_original_strict_three_attempt_rule(tmp_path, monkeypatch):
    payload = '```json\n{"memory":[]}\n```'
    llm, calls = llm_for(tmp_path, monkeypatch, [LLMResponse(payload)] * 3)
    with pytest.raises(ValueError, match='attempt=3/3'):
        invoke(llm)
    assert len(calls) == 3 and not llm.request_succeeded


@pytest.mark.parametrize('payload,accepted', [
    ('```json\n{"memory":[{"text":"fact A"}]}\n``` trailing prose', True),
    ('{"memory":[]}', True),
    ('{"wrong":[]}', True),
    ('{"memory":[{"text":"fact A"},', False),
])
def test_native_memory_receives_length_original_and_real_parse_flow(tmp_path, monkeypatch, payload, accepted):
    from mem0 import Memory
    import mem0.memory.main as native
    import socket
    def deny(*args, **kwargs):
        raise AssertionError('network forbidden')
    monkeypatch.setattr(socket.socket, 'connect', deny)
    monkeypatch.setattr(native, 'lemmatize_for_bm25', lambda text: text)
    monkeypatch.setattr(native, 'extract_entities_batch', lambda texts: [[] for _ in texts])
    monkeypatch.setattr(native, 'capture_event', lambda *a, **k: None)
    vectors, history, saved = [], [], []
    llm, calls = llm_for(tmp_path, monkeypatch, [LLMResponse(payload, finish_reason='length')])
    memory = object.__new__(Memory)
    memory.custom_instructions = None
    memory.api_version = 'v1.1'
    memory.db = SimpleNamespace(get_last_messages=lambda *a, **k: [],
        save_messages=lambda *a, **k: saved.append(True),
        batch_add_history=lambda records: history.extend(records))
    memory.embedding_model = SimpleNamespace(embed=lambda *a, **k: [.1],
        embed_batch=lambda texts, *a, **k: [[.1] for _ in texts])
    memory.vector_store = SimpleNamespace(search=lambda **k: [],
        insert=lambda **k: vectors.extend(k['payloads']))
    memory.llm = llm
    args = ([{'role': 'user', 'content': 'offline turn'}],
            {'run_id': 'offline', 'user_id': 'user_a'},
            {'run_id': 'offline', 'user_id': 'user_a'}, True)
    if accepted:
        result = memory._add_to_vector_store(*args)
        assert len(result) == (0 if payload in {'{"memory":[]}', '{"wrong":[]}'} else 1)
        assert len(vectors) == len(history) == len(result)
        assert saved == [True]
    else:
        with pytest.raises(ValueError, match='could not parse'):
            memory._add_to_vector_store(*args)
        assert not vectors and not history and not saved
    assert len(calls) == 1
    assert llm.request_succeeded is True  # This test calls native Memory directly, outside the scoped error boundary.
    assert any(row['response']['content'] == payload for row in evidence(llm))
