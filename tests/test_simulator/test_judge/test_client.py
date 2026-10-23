"""The judge client, against a mocked transport. No network in these tests."""

from __future__ import annotations

import io
import json
import socket
import threading
import time

import pytest

from simulator.judge import client as judge_client
from simulator.judge.client import (
    THINKING_TOKEN_BUDGET,
    JudgeClient,
    JudgeUnavailable,
    effective_deadline_s,
)
from simulator.judge.guards import JudgeTimeout

REPLY = json.dumps(
    {
        "verdicts": [
            {
                "question_id": "carries_an_ask",
                "verdict": "ask",
                "evidence": "Update the year field",
                "reason": "an instruction",
            }
        ]
    }
)


@pytest.fixture(autouse=True)
def judge_env(monkeypatch):
    """Every client needs both; no test may reach the real endpoint."""
    monkeypatch.setenv("MEMPA_JUDGE_API_KEY", "test-key")
    monkeypatch.setenv("MEMPA_JUDGE_URL", "http://judge.test:30001")
    # One box on purpose: these tests are about the client, not the draw.
    monkeypatch.setenv("MEMPA_JUDGE_PIN", "client tests")


class FakeResponse(io.BytesIO):
    """The SSE chat-completion frames the endpoint streams back."""

    def __init__(
        self,
        text: str,
        *,
        delay: float = 0.0,
        finish_reason: str = "stop",
        reasoning: str = "",
        reasoning_tokens: int | None = None,
    ):
        frames = [
            "data: "
            + json.dumps({"choices": [{"delta": {"reasoning": chunk}, "finish_reason": None}]})
            for chunk in (_chunks(reasoning) if reasoning else [])
        ]
        frames += [
            "data: "
            + json.dumps({"choices": [{"delta": {"content": chunk}, "finish_reason": None}]})
            for chunk in _chunks(text)
        ]
        frames.append(
            "data: "
            + json.dumps({"choices": [{"delta": {}, "finish_reason": finish_reason}]})
        )
        usage = {"prompt_tokens": 812, "completion_tokens": 44}
        if reasoning_tokens is not None:
            usage["completion_tokens"] = 44 + reasoning_tokens
            usage["completion_tokens_details"] = {"reasoning_tokens": reasoning_tokens}
        frames.append("data: " + json.dumps({"choices": [], "usage": usage}))
        frames.append("data: [DONE]")
        super().__init__(("\n\n".join(frames) + "\n\n").encode("utf-8"))
        self._delay = delay

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def __iter__(self):
        # `io.BytesIO.__iter__` returns self, so read line by line instead.
        while True:
            line = self.readline()
            if not line:
                return
            if self._delay and line.strip():
                time.sleep(self._delay)
            yield line


class StallingResponse(FakeResponse):
    """Frames arrive at once, then the model stalls past the deadline — the
    shape of a runaway generation, where the timeout lands mid-stream."""

    def __init__(self, text: str, *, before_stall: int, stall: float):
        super().__init__(text)
        self._before_stall = before_stall
        self._stall = stall

    def __iter__(self):
        sent = 0
        while True:
            line = self.readline()
            if not line:
                return
            if line.strip():
                if sent == self._before_stall:
                    time.sleep(self._stall)
                sent += 1
            yield line


def _chunks(text: str, size: int = 24):
    return [text[i : i + size] for i in range(0, len(text), size)] or [""]


@pytest.fixture
def captured(monkeypatch):
    """Records the payload of every call and replays a canned response."""
    seen: list[dict] = []

    def fake_urlopen(request, timeout=None):
        seen.append(
            {
                "url": request.full_url,
                "payload": json.loads(request.data.decode("utf-8")),
                "headers": dict(request.headers),
                "timeout": timeout,
            }
        )
        return FakeResponse(REPLY)

    monkeypatch.setattr(judge_client, "_urlopen", fake_urlopen)
    return seen


def test_the_payload_pins_the_model_and_the_generation_settings(captured):
    call = JudgeClient().generate(
        system="s", prompt="p", allowed={"carries_an_ask": ("ask", "not_applicable")}
    )
    row = captured[0]
    payload = row["payload"]
    assert payload["model"] == "gemma-4-12b-it"
    assert payload["messages"] == [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "p"},
    ]
    assert payload["temperature"] == 0
    assert payload["max_tokens"] == 256 + THINKING_TOKEN_BUDGET
    assert payload["stream"] is True
    assert payload["stream_options"] == {"include_usage": True}
    assert payload["chat_template_kwargs"] == {"enable_thinking": True}
    # Top-level, not inside chat_template_kwargs: that is the operator's fix
    # for the thinking x json_schema hang, and the place is load-bearing.
    assert payload["thinking_token_budget"] == THINKING_TOKEN_BUDGET
    assert "thinking_token_budget" not in payload["chat_template_kwargs"]
    schema = payload["response_format"]["json_schema"]["schema"]
    assert payload["response_format"]["type"] == "json_schema"
    assert schema["properties"]["verdicts"]["items"]["properties"]["verdict"][
        "enum"
    ] == ["ask", "not_applicable"]
    assert row["url"].endswith("/v1/chat/completions")
    assert row["headers"]["Authorization"] == "Bearer test-key"
    assert call.raw == REPLY
    assert (call.prompt_tokens, call.eval_tokens) == (812, 44)


def test_a_client_without_an_api_key_refuses_to_be_built(monkeypatch):
    """The endpoint takes no anonymous calls, so this fails at construction
    rather than on the first question of a live round."""
    monkeypatch.setenv("MEMPA_JUDGE_API_KEY", "")
    with pytest.raises(JudgeUnavailable, match="MEMPA_JUDGE_API_KEY"):
        JudgeClient()


def test_the_schema_pins_the_question_ids_and_their_count(captured):
    """The rule is one verdict per asked id. Said only in the parse, this model
    answers the first question of a group and stops."""
    JudgeClient().generate(
        system="s", prompt="p", allowed={"rule__q1": ("yes", "no"), "rule__q2": ("yes", "no")}
    )
    schema = captured[0]["payload"]["response_format"]["json_schema"]["schema"]
    verdicts = schema["properties"]["verdicts"]
    assert verdicts["minItems"] == 2
    assert verdicts["maxItems"] == 2
    assert verdicts["items"]["properties"]["question_id"]["enum"] == [
        "rule__q1",
        "rule__q2",
    ]
    # The count and the enum together still allow one id twice, which is what
    # `gemma-4-31b-it` did on the five-question `authorises` group: five
    # verdicts, all `__q1`. So each position carries its own question.
    assert [
        item["properties"]["question_id"]["enum"]
        for item in verdicts["prefixItems"]
    ] == [["rule__q1"], ["rule__q2"]]


def test_num_predict_follows_the_schema_width(captured):
    client = JudgeClient()
    for kind, expected in (("yes_no", 256), ("wide", 600), ("general", 2048)):
        client.generate(system="s", prompt="p", allowed={"q": ("yes", "no")}, num_predict_kind=kind)
    assert [row["payload"]["max_tokens"] for row in captured] == [
        tier + THINKING_TOKEN_BUDGET for tier in (256, 600, 2048)
    ]


def test_num_predict_is_sized_from_the_question_count(captured):
    """The tier is a floor. A two-question wide group spent all 600 on q1 and
    was cut mid-value in q2 three times over (smoke run)."""
    client = JudgeClient()
    pair = {"rule__q1": ("yes", "no"), "rule__q2": ("yes", "no")}
    client.generate(system="s", prompt="p", allowed=pair, num_predict_kind="wide")
    assert captured[-1]["payload"]["max_tokens"] == 768 + THINKING_TOKEN_BUDGET

    six = {f"rule__q{index}": ("yes", "no") for index in range(6)}
    client.generate(system="s", prompt="p", allowed=six, num_predict_kind="general")
    assert captured[-1]["payload"]["max_tokens"] == 2048 + THINKING_TOKEN_BUDGET


def test_a_group_wider_than_the_cap_gets_what_it_needs(captured):
    """`probe_fact_missing` asks one question per fact and `authorises` five
    plus its carve-outs; a cap below what they need is the truncation this
    sizing exists to stop."""
    seven = {f"probe_fact_missing:{index}": ("yes", "no") for index in range(7)}
    JudgeClient().generate(
        system="s", prompt="p", allowed=seven, num_predict_kind="general"
    )
    assert captured[-1]["payload"]["max_tokens"] == 2368 + THINKING_TOKEN_BUDGET


def test_an_explicit_num_predict_overrides_the_sizing(captured):
    """`guards.ask` sends it to widen one call that came back cut."""
    JudgeClient().generate(
        system="s",
        prompt="p",
        allowed={"q": ("yes", "no")},
        num_predict_kind="wide",
        num_predict=1200,
    )
    assert captured[-1]["payload"]["max_tokens"] == 1200 + THINKING_TOKEN_BUDGET


def test_a_reply_that_stopped_on_the_token_cap_says_so(monkeypatch, captured):
    call = JudgeClient().generate(system="s", prompt="p", allowed={"q": ("yes", "no")})
    assert call.truncated is False

    def cut_urlopen(request, timeout=None):
        return FakeResponse(REPLY, finish_reason="length")

    monkeypatch.setattr(judge_client, "_urlopen", cut_urlopen)
    cut = JudgeClient().generate(system="s", prompt="p", allowed={"q": ("yes", "no")})
    assert cut.truncated is True
    assert cut.num_predict == 256


def test_without_a_done_reason_the_token_count_says_it(captured):
    """The completion count reached the cap, which is the same statement."""
    from simulator.judge.client import JudgeCall

    assert JudgeCall("", 0.0, 10, 600, num_predict=600).truncated is True
    assert JudgeCall("", 0.0, 10, 599, num_predict=600).truncated is False
    assert JudgeCall("", 0.0, 10, 44).truncated is False
    # `completion_tokens` counts the reasoning pass too, so the count that is
    # compared against the tier is the answer's own.
    cut = JudgeCall("", 0.0, 10, 1624, num_predict=600, reasoning_tokens=1024)
    assert cut.truncated is True
    whole = JudgeCall("", 0.0, 10, 1223, num_predict=600, reasoning_tokens=1024)
    assert whole.truncated is False


def test_an_unknown_num_predict_kind_is_refused(captured):
    with pytest.raises(ValueError, match="num_predict_kind"):
        JudgeClient().generate(
            system="s", prompt="p", allowed={"q": ("yes", "no")}, num_predict_kind="huge"
        )


def test_the_model_env_var_overrides_and_nothing_else_does(monkeypatch, captured):
    monkeypatch.setenv("MEMPA_JUDGE_MODEL", "google/gemma-4-27B-it")
    assert JudgeClient().model == "google/gemma-4-27B-it"
    monkeypatch.delenv("MEMPA_JUDGE_MODEL")
    assert JudgeClient().model == "gemma-4-12b-it"


def test_the_url_env_var_is_where_the_endpoint_comes_from(monkeypatch):
    """With or without the `/v1`, and it lands there exactly once."""
    monkeypatch.setenv("MEMPA_JUDGE_URL", "http://10.0.0.3:30001/")
    assert JudgeClient().base_url == "http://10.0.0.3:30001/v1"
    monkeypatch.setenv("MEMPA_JUDGE_URL", "http://10.0.0.3:30001/v1")
    assert JudgeClient().base_url == "http://10.0.0.3:30001/v1"


def test_a_client_without_a_url_refuses_to_be_built(monkeypatch):
    """There is no default endpoint: a round with no URL set does not start,
    rather than pointing at whatever address the constant last carried."""
    monkeypatch.delenv("MEMPA_JUDGE_URL")
    # No pool either: `.env` carries one, and a pool is a URL (see test_pool).
    monkeypatch.delenv("MEMPA_JUDGE_URL_1", raising=False)
    with pytest.raises(JudgeUnavailable, match="MEMPA_JUDGE_URL"):
        JudgeClient()


def test_a_missing_model_raises_and_never_substitutes(monkeypatch):
    import urllib.error

    def fake_urlopen(request, timeout=None):
        raise urllib.error.HTTPError(
            request.full_url, 404, "Not Found", {}, io.BytesIO(b'{"error":"model not found"}')
        )

    monkeypatch.setattr(judge_client, "_urlopen", fake_urlopen)
    with pytest.raises(JudgeUnavailable, match="model not found"):
        JudgeClient().generate(system="s", prompt="p", allowed={"q": ("yes", "no")})


def test_the_endpoint_being_down_raises_rather_than_degrading(monkeypatch):
    import urllib.error

    def fake_urlopen(request, timeout=None):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(judge_client, "_urlopen", fake_urlopen)
    with pytest.raises(JudgeUnavailable, match="not reachable"):
        JudgeClient().generate(system="s", prompt="p", allowed={"q": ("yes", "no")})


def test_a_transient_status_is_retried_and_then_raises(monkeypatch):
    """Bounded backoff: `MEMPA_JUDGE_RETRIES` retries, then unavailable."""
    import urllib.error

    calls = []
    monkeypatch.setattr(judge_client, "RETRY_PAUSE_S", 0.01)

    def flaky_urlopen(request, timeout=None):
        calls.append(1)
        if len(calls) == 1:
            raise urllib.error.HTTPError(
                request.full_url, 503, "busy", {}, io.BytesIO(b"busy")
            )
        return FakeResponse(REPLY)

    monkeypatch.setattr(judge_client, "_urlopen", flaky_urlopen)
    call = JudgeClient().generate(system="s", prompt="p", allowed={"q": ("yes", "no")})
    assert len(calls) == 2
    assert call.raw == REPLY

    def always_502(request, timeout=None):
        calls.append(1)
        raise urllib.error.HTTPError(
            request.full_url, 502, "bad gateway", {}, io.BytesIO(b"bad gateway")
        )

    calls.clear()
    monkeypatch.setattr(judge_client, "_urlopen", always_502)
    with pytest.raises(JudgeUnavailable, match="HTTP 502"):
        JudgeClient().generate(system="s", prompt="p", allowed={"q": ("yes", "no")})
    # The first try plus `DEFAULT_RETRIES`.
    assert len(calls) == 1 + judge_client.DEFAULT_RETRIES


def test_the_deadline_is_passed_to_the_transport_and_read_from_the_env(monkeypatch, captured):
    monkeypatch.setenv("MEMPA_JUDGE_DEADLINE_S", "7.5")
    JudgeClient().generate(system="s", prompt="p", allowed={"q": ("yes", "no")})
    assert captured[0]["timeout"] == 7.5


def test_the_deadline_floor_covers_up_to_two_questions():
    assert effective_deadline_s(0) == 60.0
    assert effective_deadline_s(1) == 60.0
    assert effective_deadline_s(2) == 60.0


def test_the_deadline_grows_with_every_question_past_the_second():
    step = ((32.1 - 7.6) / 9) * (60.0 / 39.0)
    assert effective_deadline_s(3) == pytest.approx(60.0 + step)
    assert effective_deadline_s(10) == pytest.approx(60.0 + 8 * step)


def test_the_env_override_is_the_floor_the_question_count_builds_on(
    monkeypatch, captured
):
    monkeypatch.setenv("MEMPA_JUDGE_DEADLINE_S", "7.5")
    step = ((32.1 - 7.6) / 9) * (60.0 / 39.0)
    ten = {f"q{index}": ("yes", "no") for index in range(10)}
    JudgeClient().generate(system="s", prompt="p", allowed=ten)
    assert captured[-1]["timeout"] == pytest.approx(7.5 + 8 * step)


def test_a_socket_timeout_becomes_a_judge_timeout_carrying_the_question_ids(monkeypatch):
    def fake_urlopen(request, timeout=None):
        raise socket.timeout("timed out")

    monkeypatch.setattr(judge_client, "_urlopen", fake_urlopen)
    with pytest.raises(JudgeTimeout) as excinfo:
        JudgeClient(deadline_s=0.2).generate(
            system="s", prompt="p", allowed={"carries_an_ask": ("ask", "not_applicable"), "other": ("yes", "no")}
        )
    assert excinfo.value.event["question_ids"] == ["carries_an_ask", "other"]
    assert excinfo.value.event["seconds"] >= 0
    # Nothing arrived on this path, and the fields are still there — the reader
    # of the transcript never has to guess whether they were recorded.
    assert excinfo.value.event["partial_response"] == ""
    assert excinfo.value.event["partial_tokens"] == 0


def test_a_slow_stream_trips_the_wall_clock_deadline(monkeypatch):
    def fake_urlopen(request, timeout=None):
        return FakeResponse(REPLY, delay=0.05)

    monkeypatch.setattr(judge_client, "_urlopen", fake_urlopen)
    with pytest.raises(JudgeTimeout):
        JudgeClient(deadline_s=0.06).generate(
            system="s", prompt="p", allowed={"carries_an_ask": ("ask", "not_applicable")}
        )


def test_the_timeout_event_carries_the_text_the_model_had_emitted(monkeypatch):
    """A live run died at 80-90 s mid-stream, and the partial text was dropped,
    so nobody could see what the model was writing. The event carries it whole:
    a runaway generation is only readable untruncated."""

    def fake_urlopen(request, timeout=None):
        return StallingResponse(REPLY, before_stall=3, stall=0.35)

    monkeypatch.setattr(judge_client, "_urlopen", fake_urlopen)
    with pytest.raises(JudgeTimeout) as excinfo:
        JudgeClient(deadline_s=0.2).generate(
            system="s", prompt="p", allowed={"carries_an_ask": ("ask", "not_applicable")}
        )
    event = excinfo.value.event
    assert event["partial_tokens"] == 3
    assert event["partial_response"] == "".join(_chunks(REPLY)[:3])
    assert excinfo.value.partial_response == event["partial_response"]


def test_calls_are_serialised_one_per_process(monkeypatch):
    """Concurrency 1 is a correctness choice: a shared box took one call from
    2 seconds to 89."""
    overlaps = []
    inside = []
    lock = threading.Lock()

    def fake_urlopen(request, timeout=None):
        with lock:
            inside.append(1)
            overlaps.append(len(inside))
        time.sleep(0.02)
        with lock:
            inside.pop()
        return FakeResponse(REPLY)

    monkeypatch.setattr(judge_client, "_urlopen", fake_urlopen)
    client = JudgeClient()
    threads = [
        threading.Thread(
            target=lambda: client.generate(system="s", prompt="p", allowed={"q": ("yes", "no")})
        )
        for _ in range(4)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert max(overlaps) == 1


def test_the_deadline_does_not_count_the_wait_for_the_call_lock(monkeypatch):
    """Only bites at N>1. The Test line runs judge calls from worker threads,
    so N clocks start together and queue on the lock; with the clock started
    before the lock the wait ate the deadline and a healthy box reported a
    78-second timeout it had spent none of. The deadline is endpoint time."""

    def fake_urlopen(request, timeout=None):
        return FakeResponse(REPLY)

    monkeypatch.setattr(judge_client, "_urlopen", fake_urlopen)
    hold = 0.4
    holding = threading.Event()

    def hog():
        with judge_client._CALL_LOCK:
            holding.set()
            time.sleep(hold)

    thread = threading.Thread(target=hog)
    thread.start()
    holding.wait()
    # A deadline far shorter than the hold: started outside the lock, the first
    # frame is already past it and this raises instead of answering.
    call = JudgeClient(deadline_s=0.05).generate(
        system="s", prompt="p", allowed={"carries_an_ask": ("ask", "not_applicable")}
    )
    thread.join()
    assert call.raw == REPLY
    assert call.seconds < hold / 2


def test_every_call_logs_one_unbuffered_line(captured, capsys):
    JudgeClient().generate(system="s", prompt="p", allowed={"carries_an_ask": ("ask", "not_applicable")})
    line = capsys.readouterr().err.strip()
    assert line.startswith("judge_call model=gemma-4-12b-it ")
    for field in (
        "seconds=",
        "prompt_tokens=812",
        "completion_tokens=44",
        "cost_usd=0",
        "num_predict=256",
        "questions=carries_an_ask",
        "verdicts=carries_an_ask=ask",
        "reasoning_tokens=None",
    ):
        assert field in line


def test_an_unparsed_reply_still_logs_its_line(monkeypatch, capsys):
    def fake_urlopen(request, timeout=None):
        return FakeResponse('{"')

    monkeypatch.setattr(judge_client, "_urlopen", fake_urlopen)
    JudgeClient().generate(system="s", prompt="p", allowed={"q": ("yes", "no")})
    assert "verdicts=unparsed" in capsys.readouterr().err


def test_the_probe_sends_a_real_shaped_payload_and_checks_the_schema(captured):
    call = JudgeClient().probe()
    prompt = captured[0]["payload"]["messages"][1]["content"]
    assert "<pa_previous_turn>" in prompt and "<candidate_turn>" in prompt
    assert "[carries_an_ask]" in prompt
    assert call.raw == REPLY


def test_the_probe_fails_loudly_on_a_reply_that_is_not_the_schema(monkeypatch):
    def fake_urlopen(request, timeout=None):
        return FakeResponse('{"verdicts": []}')

    monkeypatch.setattr(judge_client, "_urlopen", fake_urlopen)
    with pytest.raises(Exception):
        JudgeClient().probe()


def test_the_reasoning_pass_is_not_part_of_the_raw_reply(monkeypatch, capsys):
    """Thinking is on, so `delta.reasoning` arrives alongside `delta.content`.
    The judge wants the verdict JSON and nothing else: the parse would choke on
    a reply with a paragraph of reasoning glued to its front."""

    def fake_urlopen(request, timeout=None):
        return FakeResponse(
            REPLY,
            reasoning="Let me weigh whether this turn puts something to the assistant.",
            reasoning_tokens=317,
        )

    monkeypatch.setattr(judge_client, "_urlopen", fake_urlopen)
    call = JudgeClient().generate(
        system="s", prompt="p", allowed={"carries_an_ask": ("ask", "not_applicable")}
    )
    assert call.raw == REPLY
    assert "weigh whether" not in call.raw
    assert call.reasoning_tokens == 317
    assert call.thinking_cut is False
    assert call.eval_tokens == 44 + 317
    assert call.answer_tokens == 44
    assert "reasoning_tokens=317" in capsys.readouterr().err


def test_a_reasoning_pass_that_hit_the_budget_says_so(monkeypatch):
    """`thinking_cut` is what the replay counts: how often the judge answered
    from a thought vLLM had to cut off at `THINKING_TOKEN_BUDGET`."""

    def fake_urlopen(request, timeout=None):
        return FakeResponse(REPLY, reasoning="x" * 80, reasoning_tokens=THINKING_TOKEN_BUDGET)

    monkeypatch.setattr(judge_client, "_urlopen", fake_urlopen)
    call = JudgeClient().generate(system="s", prompt="p", allowed={"q": ("yes", "no")})
    assert call.thinking_cut is True


def test_an_endpoint_that_sends_no_reasoning_count_leaves_the_field_none(captured):
    """The backup box sends `reasoning_tokens: null`. Absent is absent — never
    a zero that would read as a judge that did not think."""
    call = JudgeClient().generate(system="s", prompt="p", allowed={"q": ("yes", "no")})
    assert call.reasoning_tokens is None
    assert call.thinking_cut is False
    assert call.answer_tokens == 44


class ErrorFrameResponse(FakeResponse):
    """The endpoint's own error, streamed as a frame with no `choices`."""

    def __init__(self, message: str):
        super().__init__("")
        frame = json.dumps({"error": {"message": message, "type": "BadRequestError"}})
        body = self.getvalue()
        self.seek(0)
        self.truncate(0)
        self.write(("data: " + frame + "\n\n").encode("utf-8") + body)
        self.seek(0)


def test_an_error_frame_raises_with_the_server_message(monkeypatch):
    """A frame carrying `error` and no `choices` used to be dropped, and the
    call came back empty with no sign of why."""

    def erroring_urlopen(request, timeout=None):
        return ErrorFrameResponse(
            "The provided JSON schema contains features not supported by xgrammar"
        )

    monkeypatch.setattr(judge_client, "_urlopen", erroring_urlopen)
    with pytest.raises(judge_client.JudgeUnavailable) as excinfo:
        JudgeClient().generate(
            system="s", prompt="p", allowed={"carries_an_ask": ("ask", "no")}
        )
    assert "not supported by xgrammar" in str(excinfo.value)
