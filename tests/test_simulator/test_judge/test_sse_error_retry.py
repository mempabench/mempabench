"""Explicit upstream SSE errors reuse the existing transport retry budget."""
import io
import json
import os
import socket

import pytest

from simulator.judge import client
from test_simulator.test_judge.test_client import FakeResponse, REPLY


@pytest.fixture(autouse=True)
def isolated_judge(monkeypatch):
    for name in [key for key in os.environ if key.startswith('MEMPA_JUDGE_')]:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv('MEMPA_JUDGE_API_KEY', 'offline-test-key')
    monkeypatch.setenv('MEMPA_JUDGE_URL', 'http://judge.test:30001')
    monkeypatch.setenv('MEMPA_JUDGE_PIN', 'offline-sse-tests')
    monkeypatch.setenv('MEMPA_JUDGE_RETRIES', '2')
    monkeypatch.setattr(client, 'RETRY_PAUSE_S', 0)
    def denied(*args, **kwargs):
        raise AssertionError('network forbidden')
    monkeypatch.setattr(socket.socket, 'connect', denied)


def wire(error, *, choices=False, prefix=None, completed=False):
    frames = []
    if prefix:
        frames.append({'choices': [{'delta': {'content': prefix}, 'finish_reason': None}]})
    if completed:
        frames.append({'choices': [{'delta': {}, 'finish_reason': 'stop'}]})
    frame = {'error': error}
    if choices:
        frame['choices'] = [{'index': 0, 'delta': {'content': ''}, 'finish_reason': 'error'}]
    frames.append(frame)
    return io.BytesIO(('\n\n'.join('data: ' + json.dumps(f) for f in frames) + '\n\ndata: [DONE]\n').encode())


def generate():
    return client.JudgeClient().generate(system='same system', prompt='same question', allowed={'q': ('yes', 'no')})


@pytest.mark.parametrize('code', [429, 500, 502, 503, 599, 'server_error'])
@pytest.mark.parametrize('with_choices', [False, True])
def test_explicit_transient_frame_retries_same_request_and_discards_partial(monkeypatch, code, with_choices):
    calls = []
    def transport(request, timeout=None):
        calls.append((request.full_url, request.data, timeout))
        if len(calls) == 1:
            return wire({'code': code, 'message': 'upstream unavailable'}, choices=with_choices, prefix='incomplete old answer')
        return FakeResponse(REPLY)
    monkeypatch.setattr(client, '_urlopen', transport)
    result = generate()
    assert result.raw == REPLY
    assert len(calls) == 2 and calls[0] == calls[1]


@pytest.mark.parametrize('error', [
    {'code': 400}, {'code': 401}, {'code': 402}, {'code': 403}, {'code': 404},
    {'code': 408}, {'code': 422}, {'code': True}, {'code': 500.0}, {'code': '500'},
    {'code': 'unknown_error'}, {'message': 'Upstream error from Venice: Internal server error'},
    'Internal server error',
    {'code': 500, 'metadata': {'error_type': 'authentication'}},
    {'code': 'server_error', 'metadata': {'error_type': 'context_length_exceeded'}},
    {'code': 429, 'metadata': {'error_type': {'malformed': True}}},
])
def test_unknown_auth_validation_and_contradictory_errors_are_not_retried(monkeypatch, error):
    calls = []
    def transport(request, timeout=None):
        calls.append(request.data)
        return wire(error, choices=True, prefix='partial')
    monkeypatch.setattr(client, '_urlopen', transport)
    with pytest.raises(client.JudgeUnavailable, match='error frame'):
        generate()
    assert len(calls) == 1


@pytest.mark.parametrize('retries', [0, 1, 3])
def test_sse_failure_consumes_only_existing_configured_retry_count(monkeypatch, retries, capsys):
    monkeypatch.setenv('MEMPA_JUDGE_RETRIES', str(retries))
    calls = []
    def transport(request, timeout=None):
        calls.append(request.data)
        return wire({'code': 503, 'message': 'provider overloaded'})
    monkeypatch.setattr(client, '_urlopen', transport)
    with pytest.raises(client.JudgeUnavailable, match=f'gave up after {retries}'):
        generate()
    assert len(calls) == 1 + retries
    assert len(set(calls)) == 1
    stderr = capsys.readouterr().err
    assert stderr.count('judge_retry endpoint=') == retries
    assert 'offline-test-key' not in stderr


def test_completed_bad_judge_answer_is_not_retried(monkeypatch):
    calls = []
    def transport(request, timeout=None):
        calls.append(request.data)
        return FakeResponse('not a valid judge answer')
    monkeypatch.setattr(client, '_urlopen', transport)
    assert generate().raw == 'not a valid judge answer'
    assert len(calls) == 1


def test_error_after_completed_judge_answer_never_requests_second_verdict(monkeypatch):
    calls = []
    def transport(request, timeout=None):
        calls.append(request.data)
        return wire({'code': 500, 'message': 'late error'}, prefix=REPLY, completed=True)
    monkeypatch.setattr(client, '_urlopen', transport)
    with pytest.raises(client.JudgeUnavailable, match='late error'):
        generate()
    assert len(calls) == 1


def test_error_code_is_visible_without_raw_metadata_or_credentials(monkeypatch):
    monkeypatch.setenv('MEMPA_JUDGE_RETRIES', '0')
    monkeypatch.setattr(client, '_urlopen', lambda *a, **k: wire({'code': 502, 'message': 'upstream unavailable', 'metadata': {'raw': 'private upstream body'}}))
    with pytest.raises(client.JudgeUnavailable) as exc:
        generate()
    assert 'code=502' in str(exc.value)
    assert 'private upstream body' not in str(exc.value)
    assert 'offline-test-key' not in str(exc.value)


def test_missing_message_does_not_dump_metadata_or_unknown_code(monkeypatch):
    monkeypatch.setattr(client, '_urlopen', lambda *a, **k: wire({'code': 'private-token-in-invalid-code', 'metadata': {'raw': 'private upstream payload'}}))
    with pytest.raises(client.JudgeUnavailable) as exc:
        generate()
    assert 'code=unknown' in str(exc.value)
    assert 'unspecified provider error' in str(exc.value)
    assert 'private' not in str(exc.value)
