"""A declared fallback may satisfy startup; neither endpoint policy nor state is rebound."""
from types import SimpleNamespace
import socket
import pytest
from simulator.judge import session
from simulator.judge.client import JudgeClient, JudgeUnavailable

@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError('network forbidden')
    monkeypatch.setattr(socket.socket, 'connect', refuse)
    monkeypatch.setattr(socket.socket, 'connect_ex', refuse)
    monkeypatch.setattr(socket, 'create_connection', refuse)


def client():
    primary=SimpleNamespace(name='openrouter', base_url='https://primary.invalid/v1', api_key='private-a', model='google/gemma-4-31b-it')
    fallback=SimpleNamespace(name='box_b', base_url='http://fallback.invalid/v1', api_key='private-b', model='gemma-4-12b-it')
    value=SimpleNamespace(endpoints=[primary,fallback], endpoint=primary,
        preflight_endpoints=(primary,fallback), model_for=lambda e:e.model)
    return value,primary,fallback


@pytest.mark.parametrize('failure', ['catalog_timeout','missing_model','probe_error'])
def test_primary_unready_uses_exact_configured_fallback_without_rebinding(monkeypatch,capsys,failure):
    value,primary,fallback=client(); calls=[]
    def models(url,key=None):
        calls.append(('models',url))
        if url==primary.base_url:
            if failure=='catalog_timeout':raise JudgeUnavailable('catalog timeout')
            if failure=='missing_model':return {'wrong-model'}
            return {primary.model}
        assert url==fallback.base_url
        return {fallback.model}
    def probe(endpoint):
        calls.append(('probe',endpoint.base_url))
        if endpoint is primary:raise JudgeUnavailable('probe rejected')
        assert endpoint is fallback
        return SimpleNamespace(seconds=0.1)
    monkeypatch.setattr(session,'_installed_models',models);value.probe=probe
    session.preflight('live',client=value)
    assert calls[0]==('models',primary.base_url)
    assert calls[-2:]==[('models',fallback.base_url),('probe',fallback.base_url)]
    assert value.endpoint is primary
    if failure!='probe_error':assert ('probe',primary.base_url) not in calls
    error=capsys.readouterr().err
    assert 'verified_endpoint=box_b configured_bound=openrouter' in error
    assert 'private-a' not in error and 'private-b' not in error
    assert not any('bound=openrouter url=https://primary.invalid/v1 ok' in line for line in error.splitlines())


def test_all_endpoints_unready_refuses_with_original_preferred_failure(monkeypatch):
    value,primary,fallback=client(); first=JudgeUnavailable('primary timeout');seen=[]
    def models(url,key=None):
        seen.append(url)
        raise first if url==primary.base_url else JudgeUnavailable('fallback timeout')
    monkeypatch.setattr(session,'_installed_models',models)
    value.probe=lambda e:pytest.fail('unlisted endpoint cannot be probed')
    with pytest.raises(JudgeUnavailable) as caught:session.preflight('live',client=value)
    assert caught.value is first
    assert seen==[primary.base_url,fallback.base_url]


def test_fallback_wrong_model_cannot_let_unready_primary_through(monkeypatch):
    value,primary,fallback=client()
    def models(url,key=None):
        if url==primary.base_url:raise JudgeUnavailable('primary timeout')
        return {'wrong-model'}
    monkeypatch.setattr(session,'_installed_models',models)
    value.probe=lambda e:pytest.fail('wrong model cannot be probed')
    with pytest.raises(JudgeUnavailable):session.preflight('live',client=value)


def test_primary_success_still_allows_unavailable_fallback(monkeypatch,capsys):
    value,primary,fallback=client();seen=[]
    def models(url,key=None):
        if url==fallback.base_url:raise JudgeUnavailable('offline fallback')
        return {primary.model}
    monkeypatch.setattr(session,'_installed_models',models)
    value.probe=lambda e:(seen.append(e) or SimpleNamespace(seconds=.1))
    session.preflight('live',client=value)
    assert seen==[primary] and value.endpoint is primary
    error=capsys.readouterr().err
    assert 'WARNING fallback_down' in error and 'bound=openrouter' in error


def test_actual_client_preference_order_used_without_second_draw(monkeypatch):
    import os
    for key in list(os.environ):
        if key.startswith('MEMPA_JUDGE_'):monkeypatch.delenv(key)
    for n in (1,2):
        monkeypatch.setenv(f'MEMPA_JUDGE_URL_{n}',f'http://endpoint{n}.invalid/v1')
        monkeypatch.setenv(f'MEMPA_JUDGE_API_KEY_{n}','offline-key')
        monkeypatch.setenv(f'MEMPA_JUDGE_MODEL_{n}',f'model{n}')
    monkeypatch.setenv('MEMPA_JUDGE_PRIMARY','2')
    actual=JudgeClient();preferred=actual.endpoint;seen=[]
    monkeypatch.setattr(session,'_installed_models',lambda url,key=None:{e.model for e in actual.endpoints if e.base_url==url})
    monkeypatch.setattr(actual,'probe',lambda e:(seen.append(e) or SimpleNamespace(seconds=.1)))
    session.preflight('live',client=actual)
    assert seen==list(actual.preflight_endpoints)
    assert seen[0] is preferred and actual.endpoint is preferred
