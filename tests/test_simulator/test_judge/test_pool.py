"""The two-box judge pool: the weighted draw, sticky binding, failover.

No network here either — the transport is the same mocked `_urlopen` the rest
of the client tests use, keyed by the URL so one box can be made to fail.
"""

from __future__ import annotations

import os
import pathlib
import time
import urllib.error

import pytest

from simulator.judge import client as judge_client
from simulator.judge.client import (
    FAILOVER_MAX_DEADLINES,
    JudgeClient,
    JudgeUnavailable,
)
from simulator.judge.guards import JudgeTimeout
from simulator.output import SESSION_INVALID_RULES
from test_simulator.test_judge.test_client import REPLY, FakeResponse

BOX_A = "http://10.0.0.1:8000"
BOX_B = "http://10.0.0.2:30001"


@pytest.fixture
def pool_env(monkeypatch):
    """The owner's split: 1/3 to box_a, 2/3 to box_b."""
    for name in [key for key in os.environ if key.startswith("MEMPA_JUDGE_")]:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MEMPA_JUDGE_NAME_1", "box_a")
    monkeypatch.setenv("MEMPA_JUDGE_URL_1", BOX_A)
    monkeypatch.setenv("MEMPA_JUDGE_API_KEY_1", "key-a")
    monkeypatch.setenv("MEMPA_JUDGE_WEIGHT_1", "1")
    monkeypatch.setenv("MEMPA_JUDGE_NAME_2", "box_b")
    monkeypatch.setenv("MEMPA_JUDGE_URL_2", BOX_B)
    monkeypatch.setenv("MEMPA_JUDGE_API_KEY_2", "key-b")
    monkeypatch.setenv("MEMPA_JUDGE_WEIGHT_2", "2")
    return monkeypatch


@pytest.fixture
def transport(monkeypatch):
    """Records every call; `down` makes a box refuse the connection, and
    `stalled` makes it drop the call the way a box that stops answering
    mid-stream does."""
    seen: list[dict] = []
    down: set[str] = set()
    stalled: set[str] = set()

    def fake_urlopen(request, timeout=None):
        seen.append({"url": request.full_url, "headers": dict(request.headers)})
        for prefix in down:
            if request.full_url.startswith(prefix):
                raise urllib.error.URLError("connection refused")
        for prefix in stalled:
            if request.full_url.startswith(prefix):
                raise TimeoutError("deadline")
        return FakeResponse(REPLY)

    monkeypatch.setattr(judge_client, "_urlopen", fake_urlopen)
    return {"seen": seen, "down": down, "stalled": stalled}


def _generate(client: JudgeClient):
    return client.generate(
        system="",
        prompt="",
        allowed={"carries_an_ask": ("ask", "not_applicable")},
    )


# ------------------------------------------------------------------ loading


def test_a_single_url_still_makes_a_single_endpoint(monkeypatch) -> None:
    for name in [key for key in os.environ if key.startswith("MEMPA_JUDGE_")]:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MEMPA_JUDGE_URL", BOX_B)
    monkeypatch.setenv("MEMPA_JUDGE_API_KEY", "test-key")
    monkeypatch.setenv("MEMPA_JUDGE_PIN", "single-box test")
    client = JudgeClient()
    assert [endpoint.base_url for endpoint in client.endpoints] == [f"{BOX_B}/v1"]
    assert client.base_url == f"{BOX_B}/v1"


def test_the_pool_is_read_from_the_numbered_variables(pool_env) -> None:
    client = JudgeClient()
    assert [(e.name, e.base_url, e.api_key, e.weight) for e in client.endpoints] == [
        ("box_a", f"{BOX_A}/v1", "key-a", 1.0),
        ("box_b", f"{BOX_B}/v1", "key-b", 2.0),
    ]


def test_a_pool_entry_without_its_own_key_is_refused(pool_env) -> None:
    pool_env.delenv("MEMPA_JUDGE_API_KEY_2")
    with pytest.raises(JudgeUnavailable, match="MEMPA_JUDGE_API_KEY_2"):
        JudgeClient()


def test_an_explicit_base_url_beats_the_pool(pool_env) -> None:
    client = JudgeClient(base_url=BOX_B)
    assert [e.base_url for e in client.endpoints] == [f"{BOX_B}/v1"]


def test_a_single_url_in_the_environment_beats_the_pool(pool_env) -> None:
    """Pinning one box is how a replay or a bench opts out of the draw."""
    pool_env.setenv("MEMPA_JUDGE_URL", BOX_A)
    pool_env.setenv("MEMPA_JUDGE_API_KEY", "pinned-key")
    pool_env.setenv("MEMPA_JUDGE_PIN", "replay of q_handed")
    client = JudgeClient()
    assert [(e.name, e.base_url) for e in client.endpoints] == [
        ("pinned:replay_of_q_handed", f"{BOX_A}/v1")
    ]


def test_a_single_url_without_a_stated_reason_is_refused(pool_env) -> None:
    """`.env` is untracked: the pool's variables are one careless comment away
    from vanishing, and a silent fall back to one box would judge a whole round
    on that box with nobody noticing. Pinning has to declare itself."""
    pool_env.setenv("MEMPA_JUDGE_URL", BOX_A)
    pool_env.setenv("MEMPA_JUDGE_API_KEY", "pinned-key")
    with pytest.raises(JudgeUnavailable, match="MEMPA_JUDGE_PIN"):
        JudgeClient()


# --------------------------------------------------------------------- draw


def test_the_draw_follows_the_weights(pool_env) -> None:
    """Many clients in one process — the replay's shape, not a round's.

    A replay builds its clients inside one interpreter, so this is where the
    600-draw law of averages is worth asserting. It is NOT the shape that
    matters for a live round: see the subprocess test below, which is the one
    that catches a per-process collapse.
    """
    bound = [JudgeClient().endpoint.name for _ in range(600)]
    share = bound.count("box_b") / len(bound)
    assert 0.60 <= share <= 0.73, share


def test_separate_processes_do_not_all_draw_the_same_box(pool_env) -> None:
    """The runner's shape: one `run_orchestrator.py` process per session.

    A seeded draw passed the 600-in-one-interpreter test above and still sent
    every session of a live round to one box, because each session is its own
    process and each process started the same sequence. So the draw is asserted
    where it actually happens — across fresh interpreters. At weight 1:2 the
    box_a is missing from fifteen draws about twice in a thousand runs.
    """
    import subprocess
    import sys

    draw = (
        "from simulator.judge.client import JudgeClient;"
        "print(JudgeClient().endpoint.name)"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(pathlib.Path(__file__).resolve().parents[3])
    names = {
        subprocess.run(
            [sys.executable, "-c", draw], capture_output=True, text=True, env=env
        ).stdout.strip()
        for _ in range(15)
    }
    assert names == {"box_a", "box_b"}, names


def test_one_client_stays_on_the_box_it_drew(pool_env, transport) -> None:
    client = JudgeClient()
    bound = client.endpoint
    calls = [_generate(client) for _ in range(8)]
    assert {call.endpoint for call in calls} == {bound.name}
    assert {entry["url"] for entry in transport["seen"]} == {
        f"{bound.base_url}/chat/completions"
    }
    assert transport["seen"][0]["headers"]["Authorization"] == f"Bearer {bound.api_key}"


# ----------------------------------------------------------------- failover


def test_a_dead_box_hands_the_call_to_the_other_one(pool_env, transport) -> None:
    client = JudgeClient()
    dead, survivor = client._order
    transport["down"].add(dead.base_url)
    assert _generate(client).endpoint == survivor.name
    # And the client stays there: the dead box is not drawn again this session.
    transport["seen"].clear()
    assert _generate(client).endpoint == survivor.name
    assert {entry["url"] for entry in transport["seen"]} == {
        f"{survivor.base_url}/chat/completions"
    }


def test_both_boxes_down_stops_the_round(pool_env, transport) -> None:
    client = JudgeClient()
    transport["down"].update(endpoint.base_url for endpoint in client.endpoints)
    with pytest.raises(JudgeUnavailable, match="not reachable"):
        _generate(client)


def test_a_timed_out_call_is_tried_once_on_the_other_box(
    pool_env, transport, capsys
) -> None:
    """Owner ruling 2026-09-14, after nanobot acc_108: a box that drops a call
    mid-stream gets the call taken off it, not the session voided. Since
    2026-09-18 the same box is asked again first (`MEMPA_JUDGE_RETRIES`), and
    the handover is what its retries running out buys."""
    pool_env.setenv("MEMPA_JUDGE_RETRIES", "1")
    client = JudgeClient(deadline_s=5)
    dropped, survivor = client._order
    transport["stalled"].add(dropped.base_url)

    call = _generate(client)

    assert call.endpoint == survivor.name
    err = capsys.readouterr().err
    assert (
        f"judge_failover from={dropped.name} from_model=gemma-4-12b-it "
        f"to={survivor.name} to_model=gemma-4-12b-it "
        f"reason=timeout call=carries_an_ask" in err
    )
    # The judge_call line names the box that actually answered.
    assert f"endpoint={survivor.name}" in err
    # The dropped box tried twice (one plus its retry), then the survivor once,
    # and the client stays on the survivor.
    assert [entry["url"] for entry in transport["seen"]] == [
        f"{dropped.base_url}/chat/completions",
        f"{dropped.base_url}/chat/completions",
        f"{survivor.base_url}/chat/completions",
    ]
    assert client.endpoint is survivor


def test_a_timeout_on_both_boxes_still_voids_the_session(
    pool_env, transport, capsys
) -> None:
    """One failover per call and no more: the second timeout is the answer the
    session dies on, under the rule name the ledger already reads. Each box
    spends its own retries first, and neither of them answering is still a
    void rather than a walk back round the pool."""
    pool_env.setenv("MEMPA_JUDGE_RETRIES", "1")
    client = JudgeClient(deadline_s=5)
    dropped, survivor = client._order
    transport["stalled"].update(endpoint.base_url for endpoint in client.endpoints)

    with pytest.raises(JudgeTimeout) as excinfo:
        _generate(client)

    assert excinfo.value.event["event"] == "judge_timeout"
    assert "judge_timeout" in SESSION_INVALID_RULES
    assert [entry["url"] for entry in transport["seen"]] == [
        f"{dropped.base_url}/chat/completions",
        f"{dropped.base_url}/chat/completions",
        f"{survivor.base_url}/chat/completions",
        f"{survivor.base_url}/chat/completions",
    ]
    assert capsys.readouterr().err.count("judge_failover") == 1


def test_a_pinned_judge_never_fails_over(pool_env, transport) -> None:
    """Pinning is deliberate and names one box: a round pinned for a replay or
    a bench must not quietly land half its calls on the other machine."""
    # One box named by hand: the preflight probes each entry of the pool, and a
    # probe of one box is a question about that box. One retry each, so the
    # count below reads as "this box and nothing else".
    pool_env.setenv("MEMPA_JUDGE_RETRIES", "1")
    pooled = JudgeClient(deadline_s=5)
    transport["stalled"].update(endpoint.base_url for endpoint in pooled.endpoints)
    with pytest.raises(JudgeTimeout):
        pooled.generate(
            system="",
            prompt="",
            allowed={"carries_an_ask": ("ask", "not_applicable")},
            endpoint=pooled.endpoints[0],
        )
    assert [entry["url"] for entry in transport["seen"]] == [
        f"{pooled.endpoints[0].base_url}/chat/completions"
    ] * 2

    # And the whole round pinned to one box by the environment.
    pool_env.setenv("MEMPA_JUDGE_URL", BOX_A)
    pool_env.setenv("MEMPA_JUDGE_API_KEY", "pinned-key")
    pool_env.setenv("MEMPA_JUDGE_PIN", "endpoint bench")
    client = JudgeClient(deadline_s=5)
    transport["seen"].clear()

    with pytest.raises(JudgeTimeout):
        _generate(client)

    assert [entry["url"] for entry in transport["seen"]] == [
        f"{BOX_A}/v1/chat/completions"
    ] * 2


def test_a_call_past_its_failover_window_is_not_retried(
    pool_env, monkeypatch, capsys
) -> None:
    """The window is `FAILOVER_MAX_DEADLINES` of the call's own deadline. A
    call that has already run through it is the runaway the void is for, and
    handing it to the other box would spend a third deadline on it."""
    client = JudgeClient(deadline_s=0.05)
    bound = client.endpoint
    seen: list[str] = []

    def stall(request, timeout=None):
        seen.append(request.full_url)
        time.sleep(FAILOVER_MAX_DEADLINES * 0.05 + 0.25)
        raise TimeoutError("deadline")

    monkeypatch.setattr(judge_client, "_urlopen", stall)
    with pytest.raises(JudgeTimeout):
        _generate(client)

    assert seen == [f"{bound.base_url}/chat/completions"]
    assert "judge_failover" not in capsys.readouterr().err
    assert client.endpoint is bound


# ------------------------------------------------------------------ logging


def test_the_log_line_names_the_box(pool_env, transport, capsys) -> None:
    client = JudgeClient()
    call = _generate(client)
    assert f"endpoint={call.endpoint}" in capsys.readouterr().err


# ---------------------------------------------- primary, models, WAN retry


def test_primary_puts_one_box_first_in_every_draw(pool_env, transport) -> None:
    """Owner ruling 2026-09-16: PRIMARY replaces the dice with an order."""
    pool_env.setenv("MEMPA_JUDGE_PRIMARY", "2")
    for _ in range(20):
        client = JudgeClient()
        assert [e.name for e in client._order] == ["box_b", "box_a"]

    pool_env.setenv("MEMPA_JUDGE_PRIMARY", "1")
    for _ in range(20):
        assert [e.name for e in JudgeClient()._order] == ["box_a", "box_b"]


def test_primary_unset_leaves_the_weighted_draw_alone(pool_env, transport) -> None:
    """The pool that existed before this is the pool that still runs."""
    pool_env.delenv("MEMPA_JUDGE_PRIMARY", raising=False)
    firsts = {JudgeClient()._order[0].name for _ in range(200)}
    assert firsts == {"box_a", "box_b"}


def test_primary_must_name_a_pool_entry(pool_env, transport) -> None:
    pool_env.setenv("MEMPA_JUDGE_PRIMARY", "3")
    with pytest.raises(JudgeUnavailable, match="names no pool entry"):
        JudgeClient()
    pool_env.setenv("MEMPA_JUDGE_PRIMARY", "primary")
    with pytest.raises(JudgeUnavailable, match="MEMPA_JUDGE_PRIMARY"):
        JudgeClient()


def test_each_box_is_called_with_its_own_model(pool_env, transport, capsys) -> None:
    """A hosted primary and a local fallback do not serve the same model id."""
    import json as _json

    pool_env.setenv("MEMPA_JUDGE_MODEL_1", "openai/gpt-oss-120b")
    pool_env.setenv("MEMPA_JUDGE_MODEL_2", "gemma-4-12b-it")
    pool_env.setenv("MEMPA_JUDGE_PRIMARY", "1")
    bodies: list[dict] = []

    def recording_urlopen(request, timeout=None):
        bodies.append(_json.loads(request.data.decode("utf-8")))
        return FakeResponse(REPLY)

    pool_env.setattr(judge_client, "_urlopen", recording_urlopen)
    client = JudgeClient()
    assert client.model_for(client.endpoint) == "openai/gpt-oss-120b"
    _generate(client)
    assert bodies[0]["model"] == "openai/gpt-oss-120b"
    assert "model=openai/gpt-oss-120b" in capsys.readouterr().err

    # No per-endpoint model: the client-wide one, exactly as before.
    pool_env.delenv("MEMPA_JUDGE_MODEL_1")
    bodies.clear()
    client = JudgeClient(model="gemma-4-12b-it")
    _generate(client)
    assert bodies[0]["model"] == "gemma-4-12b-it"


def test_a_5xx_is_retried_to_the_bound_and_then_fails_over(
    pool_env, transport, capsys, monkeypatch
) -> None:
    """503 on the primary: retried in place, then the fallback takes the call."""
    import io as _io
    import urllib.error as _ue

    monkeypatch.setattr(judge_client, "RETRY_PAUSE_S", 0.001)
    monkeypatch.setenv("MEMPA_JUDGE_RETRIES", "3")
    monkeypatch.setenv("MEMPA_JUDGE_PRIMARY", "1")
    monkeypatch.setenv("MEMPA_JUDGE_MODEL_1", "openai/gpt-oss-120b")
    seen: list[str] = []

    def flaky(request, timeout=None):
        seen.append(request.full_url)
        if request.full_url.startswith(BOX_A):
            raise _ue.HTTPError(
                request.full_url, 503, "busy", {}, _io.BytesIO(b"busy")
            )
        return FakeResponse(REPLY)

    monkeypatch.setattr(judge_client, "_urlopen", flaky)
    client = JudgeClient()
    call = _generate(client)

    # Four tries on the primary (one plus three retries), then the fallback.
    assert seen.count(f"{BOX_A}/v1/chat/completions") == 4
    assert call.endpoint == "box_b"
    err = capsys.readouterr().err
    assert "judge_retry endpoint=box_a model=openai/gpt-oss-120b attempt=1/3" in err
    assert "judge_retry endpoint=box_a model=openai/gpt-oss-120b attempt=3/3" in err
    assert "judge_failover from=box_a" in err and "reason=unavailable" in err
    # Sticky: the client stays on the fallback for the rest of the session.
    assert client.endpoint.name == "box_b"


def test_the_retry_count_is_bounded_by_the_env(
    pool_env, transport, monkeypatch
) -> None:
    import io as _io
    import urllib.error as _ue

    monkeypatch.setattr(judge_client, "RETRY_PAUSE_S", 0.001)
    monkeypatch.setenv("MEMPA_JUDGE_PRIMARY", "1")
    for retries in (0, 1, 5):
        monkeypatch.setenv("MEMPA_JUDGE_RETRIES", str(retries))
        seen: list[str] = []

        def always_503(request, timeout=None):
            seen.append(request.full_url)
            raise _ue.HTTPError(
                request.full_url, 503, "busy", {}, _io.BytesIO(b"busy")
            )

        monkeypatch.setattr(judge_client, "_urlopen", always_503)
        with pytest.raises(JudgeUnavailable, match="HTTP 503"):
            _generate(JudgeClient())
        # Both boxes, each tried once plus its retries, over every pass the
        # call makes at the pool.
        assert len(seen) == judge_client.POOL_PASSES * 2 * (1 + retries)


def test_an_empty_body_is_a_transport_failure_not_an_answer(
    pool_env, transport, monkeypatch
) -> None:
    """A 200 that streamed nothing is the WAN dropping the call, not a verdict."""
    monkeypatch.setattr(judge_client, "RETRY_PAUSE_S", 0.001)
    monkeypatch.setenv("MEMPA_JUDGE_RETRIES", "2")
    monkeypatch.setenv("MEMPA_JUDGE_PRIMARY", "1")
    seen: list[str] = []

    def empty_on_the_primary(request, timeout=None):
        seen.append(request.full_url)
        if request.full_url.startswith(BOX_A):
            return FakeResponse("")
        return FakeResponse(REPLY)

    monkeypatch.setattr(judge_client, "_urlopen", empty_on_the_primary)
    call = _generate(JudgeClient())
    assert seen.count(f"{BOX_A}/v1/chat/completions") == 3
    assert call.endpoint == "box_b"


def test_a_dropped_connection_mid_stream_is_retried(
    pool_env, transport, monkeypatch
) -> None:
    """The TLS EOF that killed WAN lanes under the one-retry policy."""
    import ssl as _ssl

    monkeypatch.setattr(judge_client, "RETRY_PAUSE_S", 0.001)
    monkeypatch.setenv("MEMPA_JUDGE_RETRIES", "3")
    monkeypatch.setenv("MEMPA_JUDGE_PRIMARY", "1")
    seen: list[str] = []

    def tls_eof_once(request, timeout=None):
        seen.append(request.full_url)
        if len(seen) == 1:
            raise _ssl.SSLEOFError("EOF occurred in violation of protocol")
        return FakeResponse(REPLY)

    monkeypatch.setattr(judge_client, "_urlopen", tls_eof_once)
    call = _generate(JudgeClient())
    assert len(seen) == 2
    # Retried in place: the primary answered the second try, no failover.
    assert call.endpoint == "box_a"


def test_a_pinned_client_never_fails_over(monkeypatch, transport) -> None:
    """A single URL pins one box; a pinned round has nowhere to go."""
    import io as _io
    import urllib.error as _ue

    monkeypatch.setattr(judge_client, "RETRY_PAUSE_S", 0.001)
    monkeypatch.setenv("MEMPA_JUDGE_RETRIES", "1")
    for index in (1, 2):
        monkeypatch.delenv(f"MEMPA_JUDGE_URL_{index}", raising=False)
    monkeypatch.setenv("MEMPA_JUDGE_URL", BOX_A)
    monkeypatch.setenv("MEMPA_JUDGE_API_KEY", "key-a")
    monkeypatch.setenv("MEMPA_JUDGE_PIN", "gold replay")
    seen: list[str] = []

    def always_503(request, timeout=None):
        seen.append(request.full_url)
        raise _ue.HTTPError(request.full_url, 503, "busy", {}, _io.BytesIO(b"busy"))

    monkeypatch.setattr(judge_client, "_urlopen", always_503)
    client = JudgeClient()
    assert len(client.endpoints) == 1
    with pytest.raises(JudgeUnavailable, match="HTTP 503"):
        _generate(client)
    # One box, one try plus its one retry, and no second URL in the log.
    assert seen == [f"{BOX_A}/v1/chat/completions"] * 2


def test_a_single_url_still_needs_the_pin(monkeypatch) -> None:
    for index in (1, 2):
        monkeypatch.delenv(f"MEMPA_JUDGE_URL_{index}", raising=False)
    monkeypatch.delenv("MEMPA_JUDGE_PIN", raising=False)
    monkeypatch.setenv("MEMPA_JUDGE_URL", BOX_A)
    monkeypatch.setenv("MEMPA_JUDGE_API_KEY", "key-a")
    with pytest.raises(JudgeUnavailable, match="MEMPA_JUDGE_PIN"):
        JudgeClient()


def test_a_fallback_that_is_down_does_not_block_the_preflight(
    pool_env, capsys, monkeypatch
) -> None:
    """Owner: the fallback is a fallback. Down, it is a warning, not a stop."""
    from simulator.judge import session as judge_session

    monkeypatch.setenv("MEMPA_JUDGE_PRIMARY", "1")
    monkeypatch.setenv("MEMPA_JUDGE_MODEL_1", "openai/gpt-oss-120b")
    monkeypatch.setenv("MEMPA_JUDGE_MODEL_2", "gemma-4-12b-it")

    def models_for(base_url, api_key=None):
        if base_url.startswith(BOX_A):
            return {"openai/gpt-oss-120b"}
        raise JudgeUnavailable(f"the judge endpoint is not reachable at {base_url}")

    monkeypatch.setattr(judge_session, "_installed_models", models_for)
    monkeypatch.setattr(judge_client, "_urlopen", lambda r, timeout=None: FakeResponse(REPLY))

    judge_session.preflight("on", client=JudgeClient())
    err = capsys.readouterr().err
    assert "endpoint=box_b" in err and "WARNING fallback_down" in err
    assert "bound=box_a" in err and "ok" in err

    # The bound box down is still the hard stop it has always been.
    def nothing_reachable(base_url, api_key=None):
        raise JudgeUnavailable(f"the judge endpoint is not reachable at {base_url}")

    monkeypatch.setattr(judge_session, "_installed_models", nothing_reachable)
    with pytest.raises(JudgeUnavailable):
        judge_session.preflight("on", client=JudgeClient())


# ------------------------------------- timeout retry, and back to the primary


def test_a_timeout_is_retried_on_the_same_box_before_the_failover(
    pool_env, transport, monkeypatch, capsys
) -> None:
    """Owner ruling 2026-09-18, after the live run that died on one slow
    OpenRouter call: a timeout is a transient failure of that box like a 503
    is, so it is retried in place first and only then handed over."""
    monkeypatch.setattr(judge_client, "RETRY_PAUSE_S", 0.001)
    monkeypatch.setenv("MEMPA_JUDGE_RETRIES", "2")
    monkeypatch.setenv("MEMPA_JUDGE_PRIMARY", "1")
    seen: list[str] = []

    def stalls_on_the_primary(request, timeout=None):
        seen.append(request.full_url)
        if request.full_url.startswith(BOX_A):
            raise TimeoutError("deadline")
        return FakeResponse(REPLY)

    monkeypatch.setattr(judge_client, "_urlopen", stalls_on_the_primary)
    call = _generate(JudgeClient(deadline_s=5))

    # One try plus its two retries on the box that timed out, then the fallback.
    assert seen.count(f"{BOX_A}/v1/chat/completions") == 3
    assert call.endpoint == "box_b"
    err = capsys.readouterr().err
    assert "judge_retry endpoint=box_a" in err and "attempt=2/2" in err
    assert "judge_failover from=box_a" in err and "reason=timeout" in err


def test_a_retry_gets_its_own_deadline_not_the_leftovers(
    pool_env, transport, monkeypatch
) -> None:
    """The bug of 2026-09-18: the retries shared the call's one deadline, so a
    first attempt that spent it left no room for any of them and the box was
    called unavailable after 0 retries. Every attempt gets a whole deadline."""
    monkeypatch.setattr(judge_client, "RETRY_PAUSE_S", 0.05)
    monkeypatch.setenv("MEMPA_JUDGE_RETRIES", "3")
    monkeypatch.setenv("MEMPA_JUDGE_PRIMARY", "1")
    seen: list[str] = []

    def slow_then_good(request, timeout=None):
        seen.append(request.full_url)
        if len(seen) == 1:
            time.sleep(0.25)
            raise urllib.error.HTTPError(
                request.full_url, 503, "busy", {}, __import__("io").BytesIO(b"busy")
            )
        return FakeResponse(REPLY)

    monkeypatch.setattr(judge_client, "_urlopen", slow_then_good)
    call = _generate(JudgeClient(deadline_s=0.3))

    assert len(seen) == 2
    assert call.endpoint == "box_a"


def test_every_fallback_down_comes_back_to_the_primary(
    pool_env, transport, monkeypatch, capsys
) -> None:
    """Owner ruling 2026-09-18: the pool walked to its end used to be the end
    of the run. The primary gets one more pass first."""
    monkeypatch.setattr(judge_client, "RETRY_PAUSE_S", 0.001)
    monkeypatch.setenv("MEMPA_JUDGE_RETRIES", "0")
    monkeypatch.setenv("MEMPA_JUDGE_PRIMARY", "1")
    seen: list[str] = []

    def primary_recovers(request, timeout=None):
        seen.append(request.full_url)
        if request.full_url.startswith(BOX_B):
            raise urllib.error.URLError("connection refused")
        if len([url for url in seen if url.startswith(BOX_A)]) == 1:
            raise urllib.error.URLError("connection refused")
        return FakeResponse(REPLY)

    monkeypatch.setattr(judge_client, "_urlopen", primary_recovers)
    client = JudgeClient(deadline_s=5)
    call = _generate(client)

    assert [url for url in seen] == [
        f"{BOX_A}/v1/chat/completions",
        f"{BOX_B}/v1/chat/completions",
        f"{BOX_A}/v1/chat/completions",
    ]
    assert call.endpoint == "box_a"
    assert client.endpoint.name == "box_a"
    assert "judge_failover from=box_b" in capsys.readouterr().err


def test_the_walk_back_to_the_primary_is_bounded(
    pool_env, transport, monkeypatch
) -> None:
    """Coming back round must not become a loop: the pool is walked a fixed
    number of times and then the judge is unavailable, as before."""
    monkeypatch.setattr(judge_client, "RETRY_PAUSE_S", 0.001)
    monkeypatch.setenv("MEMPA_JUDGE_RETRIES", "0")
    monkeypatch.setenv("MEMPA_JUDGE_PRIMARY", "1")
    seen: list[str] = []

    def all_down(request, timeout=None):
        seen.append(request.full_url)
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(judge_client, "_urlopen", all_down)
    with pytest.raises(JudgeUnavailable, match="not reachable"):
        _generate(JudgeClient(deadline_s=5))

    assert len(seen) == judge_client.POOL_PASSES * 2
