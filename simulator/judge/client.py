"""vLLM chat-completions client for the judge (contract §6).

An OpenAI-compatible endpoint, pinned model, temperature 0, the §3 schema
passed as `response_format: json_schema` so the sampler honours it, one call at
a time per process, a wall-clock deadline, and one unbuffered log line per call.
The pool is one primary endpoint and its fallbacks, each carrying its own model
id, and which one is primary is an env decision (`MEMPA_JUDGE_PRIMARY`) rather
than a constant in here. There is no degrade to a regex, and if no endpoint in
the pool answers, the run does not start.

Transport note: every call streams. The reply arrives as SSE `data:` frames
whose `choices[0].delta.content` are concatenated; the last frame before
`data: [DONE]` carries `usage`, and `finish_reason` says whether the generation
stopped on the model finishing or on the token cap. Thinking is on, so the same
frames also carry `choices[0].delta.reasoning` — the reasoning pass, which is
never appended to `raw`: the judge wants the verdict JSON and nothing else.

Proxy note: this client opens the endpoint with the proxy handlers empty. A
local HTTP proxy in the environment (`HTTP_PROXY`) sent every judge call to a
proxy that answered 502 with an empty body after 81 s — which reads exactly
like a dead endpoint, and is not one.
"""

from __future__ import annotations

import http.client
import json
import os
import random
import ssl
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from simulator.judge.guards import ASK_GATE_VALUES, JudgeTimeout, strip_fences

#: The judge endpoint comes from `MEMPA_JUDGE_URL` and from nowhere else: there
#: is no default, because a hardcoded address survives the host it was written
#: for and points a round at a machine nobody is watching. Unset or empty is a
#: hard error at construction, the way a missing key is. It takes the URL with
#: or without the trailing `/v1` — the constructor normalises it so the `/v1` is
#: there exactly once. It is read in the constructor the way
#: `MEMPA_JUDGE_DEADLINE_S` is, so live sessions, the replay and the preflight
#: all pick it up without a call-site change. The preflight line prints the
#: base_url in use, so a round pointed at the wrong host is visible in the first
#: line of the log.

#: Pinned, because it is the one the replay's gold table is measured against.
#: `MEMPA_JUDGE_MODEL` overrides it for a deliberate model swap, which per §6
#: means re-running the replay. There is no second candidate and no fallback.
DEFAULT_MODEL = "gemma-4-12b-it"

#: Thinking is on: the owner's ruling is that the judge reasons before it
#: answers, and that every measurement is taken under this one production
#: config. The deployment's `gemma4` reasoning parser puts that pass in
#: `delta.reasoning`, separate from the answer, so the schema still constrains
#: the answer alone.
CHAT_TEMPLATE_KWARGS = {"enable_thinking": True}

#: Thinking and `json_schema` together hang on this deployment: the reasoning
#: pass never ends, so the answer never starts. The operator's fix is this
#: per-request budget, sent as a **top-level** field and not inside
#: `chat_template_kwargs`: when the budget is reached vLLM forces the
#: end-of-thinking token and only then applies the schema. Operator's
#: measurement on the longest judge sample at 4 concurrent: 512 gave P95 29 s,
#: 1024 gave P95 37 s — 1024 buys the longer reasoning for 8 s of tail. There
#: is no env override: one production config, no knobs.
THINKING_TOKEN_BUDGET = 1024

#: The endpoint answers 429 or a 5xx while a replica is busy or restarting, and
#: a WAN endpoint adds its own transport failures on top: a refused or reset
#: connection, a TLS EOF mid-stream, a 200 with an empty body. All of them are
#: the same thing from here — the call never happened — and all of them are
#: retried in place before the endpoint is called unavailable and the client
#: fails over. `RETRY_STATUSES` keeps its name and its two named codes; every
#: other 5xx joins them through `_retryable_status`, because over the WAN the
#: gateway picks the code and the list was never going to be complete.
RETRY_STATUSES = (429, 502, 503)

#: How many retries one call on one endpoint gets, and the pause before each.
#: `MEMPA_JUDGE_RETRIES` overrides the count; the backoff is doubled per retry
#: from `RETRY_PAUSE_S` and is not a knob. Three retries at 2/4/8 s covers the
#: transient TLS EOF and 429 bursts that killed whole lanes under the previous
#: one-retry policy, and costs 14 s before the fallback box is tried. A timeout
#: is retried on the same count but with no pause at all (owner ruling
#: 2026-09-18): the attempt that timed out has already waited a whole deadline
#: on that box, so a backoff on top of it buys nothing.
DEFAULT_RETRIES = 3
RETRY_PAUSE_S = 2.0

#: How many times a call walks the pool before the judge is unavailable. Two,
#: because one was the live run of 2026-09-18: the primary timed out, both
#: fallback boxes were unreachable, and the walk ran off the end of the list
#: with the whole round exiting on it — while the primary was answering again
#: seconds later. So the end of the list comes back round to the head once more
#: (owner ruling 2026-09-18), and then stops: a pool walked twice over is a
#: judge that is genuinely not there, and looping on it would hang the round
#: instead of failing it.
POOL_PASSES = 2

#: The default is the measured maximum under load plus headroom (owner rule).
#: Measured on the smoke runs of 2026-09-09, one lane, model resident: a single
#: question 3.7-11 s, a 1748-token two-question pair 23-39 s under load average
#: 5.3 (33 s on acc_003), the preflight probe 10-13 s. 15 s covered only the
#: single questions and voided sessions on any round that forgot the env
#: override; 60 s = measured max 39 s + headroom. `MEMPA_JUDGE_DEADLINE_S`
#: still overrides per round. It bounds ONE attempt, not the whole call: every
#: retry starts a fresh one (owner ruling 2026-09-18), so a single slow call can
#: no longer eat the retries that were meant to cover it.
DEFAULT_DEADLINE_S = 60.0

#: The floor covers up to two questions, and every question above that adds the
#: measured mean increment per question. From 主-解读's step-0 gate_cue replay
#: 2026-09-09 (deadline 60, load average 11-14): one question 7.6 s mean, ten
#: 32.1 s, so (32.1 - 7.6) / 9 s per extra question.
DEADLINE_PER_QUESTION_S = (32.1 - 7.6) / 9

#: The floor's own headroom ratio, reused for the per-question term so the two
#: halves of the deadline are equally generous: 60 s over the measured 39 s max.
DEADLINE_HEADROOM = DEFAULT_DEADLINE_S / 39.0

#: Up to this many questions the floor already covers (the 39 s max it was set
#: from was a two-question pair).
DEADLINE_FLOOR_QUESTIONS = 2

#: A call that timed out mid-stream is given to the other box once (owner
#: ruling 2026-09-14, after nanobot acc_108 voided on a workstation that
#: dropped mid-call and the whole session was rerun pinned to the other box).
#: The window is measured in the call's own deadlines, because that deadline is
#: the only per-call timeout there is: a stream that dropped has already spent
#: one of them by definition — the socket's timeout IS the deadline, so the
#: timeout lands somewhere between one and two of them — and a call that has
#: burned through two is the runaway the void exists for. The retry runs on its
#: own fresh deadline, so a failed-over call costs at most three where the void
#: it replaces costs a rerun of the whole session.
#:
#: The same window bounds the in-place retry of a timeout (`_stream`): one
#: attempt that ran through two of its own deadlines before dropping is the
#: runaway this window names, and asking that box again is not what it needs.
FAILOVER_MAX_DEADLINES = 2.0


def effective_deadline_s(n_questions: int, floor: float = DEFAULT_DEADLINE_S) -> float:
    """`floor` plus the measured cost of every question past the second.

    `floor + max(0, n - 2) * ((32.1 - 7.6) / 9) * (60 / 39)`. The floor and its
    39 s basis are `DEFAULT_DEADLINE_S`'s own; the increment and its headroom
    ratio are the two constants above. `MEMPA_JUDGE_DEADLINE_S` sets the floor
    when it is set, so an override is still the smallest deadline a call gets.
    """
    extra = max(0, n_questions - DEADLINE_FLOOR_QUESTIONS)
    return floor + extra * DEADLINE_PER_QUESTION_S * DEADLINE_HEADROOM

#: `num_predict` follows the schema's width (§6). 300/400 truncated the wide
#: two-step schema mid-string; 900 truncated 21% of general-form replies.
NUM_PREDICT = {"yes_no": 256, "wide": 600, "general": 2048}

#: A tier is not enough on its own: the wide tier's 600 is a whole-call budget,
#: and a two-question call whose q1 quoted a long span spent all 600 on q1's
#: evidence and reason and was cut mid-value in q2 (three times over, the smoke
#: run of 2026-09-09). So the tier is a floor and the group's own width
#: sets the rest: one verdict with a verbatim evidence span and a one-sentence
#: reason fits comfortably in 320 tokens, plus 128 for the wrapper.
PER_QUESTION_NUM_PREDICT = 320
NUM_PREDICT_HEADROOM = 128


def effective_num_predict(kind: str, n_questions: int) -> int:
    """The tier, widened to what this many questions actually need.

    A single question keeps its tier: 256/600/2048 are the measured sizes of a
    one-question call, and the replies to those run 76-186 tokens. From two
    questions up the group's own width wins where it is larger, capped at the
    general tier so an ordinary group cannot ask for an unbounded generation —
    except where the questions alone exceed that cap (`authorises`' five plus
    its carve-outs, `probe_fact_missing`'s one per fact), because a cap below
    what the group needs is the truncation this rule exists to stop.
    """
    if kind not in NUM_PREDICT:
        raise ValueError(
            f"num_predict_kind must be one of {sorted(NUM_PREDICT)}, got {kind!r}"
        )
    tier = NUM_PREDICT[kind]
    if n_questions <= 1:
        return tier
    questions = PER_QUESTION_NUM_PREDICT * n_questions
    needed = max(tier, questions + NUM_PREDICT_HEADROOM)
    cap = NUM_PREDICT["general"]
    return needed if questions > cap else min(needed, cap)

#: The §3 verdict schema, passed as `response_format` rather than asked for in
#: prose. Unconstrained, this model answers a group of one with a fenced,
#: unwrapped single verdict (step-0 probe, 2026-09-10), so the constraint is
#: what makes the reply parseable at all.
VERDICT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "verdicts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "question_id": {"type": "string"},
                    "verdict": {
                        "type": "string",
                        "enum": ["obeyed", "violated", "ask", "not_applicable", "yes", "no"],
                    },
                    "evidence": {"type": "string"},
                    # `minLength` so constrained decoding cannot emit "": ~8
                    # live answers came back with an empty `evidence` AND an
                    # empty `reason`, schema-legal and unauditable (regress23
                    # read, finding 2), and `withheld_fact_before_ask` reads
                    # the first word of `reason` as the slot noun.
                    "reason": {"type": "string", "minLength": 1},
                },
                "required": ["question_id", "verdict", "evidence", "reason"],
            },
        }
    },
    "required": ["verdicts"],
}


def verdict_schema(allowed: Mapping[str, Sequence[str]]) -> dict[str, Any]:
    """§3's "one for one with the ids in the request", stated in the schema.

    Without the count, this model answers the first question of a group and
    stops: 21 of 57 two-step and per-fact calls came back well-formed, under
    the token cap, and one verdict short. §3 already calls that a schema
    failure, so it belongs in `response_format` where the sampler can honour
    it, rather than only in the parse that rejects it afterwards.

    The verdict enum is the union over the group's questions. Every group in
    the catalogue keeps to one vocabulary — all its questions are sub-questions,
    or it is a single whole-rule question, or it is the ask gate — so the union
    is exact for each one. A sub-question may add `not_applicable`
    (`guards.YES_NO_NA_VALUES`); that only widens the union, and
    `parse_verdicts` still checks each question against its own set, which is
    what catches a group that mixes vocabularies outright.

    The count and the enum together still let one id answer twice, and
    `gemma-4-31b-it` does exactly that: on the five-question `authorises` group
    it returned five verdicts, all of them `__q1`, one per clause of
    `beat_avoid` — schema-legal, under the cap, and missing four of the five
    questions. At temperature 0 all three of `guards.ask`'s attempts come back
    identical, so the group can never parse and the session voids
    (`judge_unparseable`). "One for one with the ids in the request" is a
    *tuple*, not a bag of the right size, so it is stated as one: position `i`
    carries question `i` and nothing else, with that question's own vocabulary.
    `items` keeps the bag form underneath, so a sampler that ignores
    `prefixItems` is no worse off than before this existed.
    """
    if not allowed:
        raise ValueError(
            "a judge call needs at least one question; an empty group makes no "
            "call (see simulator.judge.guards.ask)"
        )
    schema = json.loads(json.dumps(VERDICT_SCHEMA))
    verdicts = schema["properties"]["verdicts"]
    verdicts["minItems"] = len(allowed)
    verdicts["maxItems"] = len(allowed)
    verdicts["items"]["properties"]["question_id"] = {
        "type": "string",
        "enum": list(allowed),
    }
    union = list(dict.fromkeys(value for values in allowed.values() for value in values))
    verdicts["items"]["properties"]["verdict"] = {"type": "string", "enum": union}
    item = verdicts["items"]
    verdicts["prefixItems"] = [
        {
            "type": "object",
            "properties": {
                **item["properties"],
                "question_id": {"type": "string", "enum": [question_id]},
                "verdict": {"type": "string", "enum": list(values)},
            },
            "required": list(item["required"]),
        }
        for question_id, values in allowed.items()
    ]
    return schema

#: Two boxes can serve the same pinned model (a hosted endpoint and a local
#: server, say), so a round can spread its calls over both. The pool
#: is on when `MEMPA_JUDGE_URL_1` is set, and off otherwise: a process with
#: only `MEMPA_JUDGE_URL` behaves exactly as it did before this existed, which
#: is what the replay and the endpoint bench need in order to measure one box.
#: Entries are numbered from 1 and read until the first gap:
#: `MEMPA_JUDGE_URL_<n>`, `MEMPA_JUDGE_API_KEY_<n>` (each box has its own key),
#: `MEMPA_JUDGE_WEIGHT_<n>` (share of the traffic, default 1),
#: `MEMPA_JUDGE_NAME_<n>` (what the log line calls it, default `node<n>`),
#: `MEMPA_JUDGE_MODEL_<n>` (the model id this box serves, default the
#: client-wide `MEMPA_JUDGE_MODEL`). The per-endpoint model is what lets one
#: pool hold boxes that do not serve the same weights at all — a hosted
#: `openai/gpt-oss-120b` and a local `gemma-4-12b-it` — so the fallback is a
#: different model on a different machine rather than a second copy of the
#: first. Every log line that names an endpoint names its model beside it.
#:
#: `MEMPA_JUDGE_PRIMARY=<n>` turns the draw off and makes entry `<n>` the box
#: every session starts on, the rest falling in behind it in index order as
#: fallbacks: one primary that serves the measured standard, and boxes below it
#: that only ever see traffic the primary could not take. Which entry that is
#: belongs in the environment, not in here — the two boxes are not the same
#: judge, so naming a default in the code would be naming a standard. Unset,
#: the weighted draw below is what happens, unchanged.
#:
#: The draw is per client and not per call: one client is one session (and one
#: replay), so a conversation is judged end to end by the same box. The split
#: comes out over sessions, and a verdict that looks odd belongs to exactly one
#: machine instead of to whichever one the dice picked that turn.


#: The draw is not seeded. It was, once, and the seed could only ever do harm
#: here: the runners start one `run_orchestrator.py` per session, so a seed
#: shared by those processes hands every session the same first draw and the
#: whole round lands on one box, with nothing in the log to say so. The
#: reproducibility it was supposed to buy — "fix the order the sessions are
#: handed out in" — was never obtainable either, because with parallel lanes
#: the order clients are constructed in is itself nondeterministic. If the
#: assignment ever has to be reproducible, the honest form is a function of the
#: session id, so that one session always lands on the same box no matter when
#: it runs; that is a different mechanism from a sequence seed, and nothing
#: asks for it today.
_DRAW = random.Random()


@dataclass(frozen=True)
class JudgeEndpoint:
    """One box in the pool. `weight` is a share, not a probability.

    `model` empty means "whatever the client is pinned to"; a box that serves
    its own model id carries it here, and `JudgeClient.model_for` resolves the
    two.
    """

    name: str
    base_url: str
    api_key: str
    weight: float = 1.0
    model: str = ""


def load_endpoints(
    *, base_url: str | None = None, api_key: str | None = None
) -> list[JudgeEndpoint]:
    """The pool if `MEMPA_JUDGE_URL_1` is set, else the single endpoint.

    An explicit `base_url` passed to the client wins over both: a caller that
    names a box means that box.
    """
    if base_url:
        return [
            JudgeEndpoint(
                name="pinned",
                base_url=normalise_base_url(base_url),
                api_key=(api_key or os.environ.get("MEMPA_JUDGE_API_KEY", "")).strip(),
            )
        ]
    #: A single `MEMPA_JUDGE_URL` wins over the pool, and is not merged into
    #: it: naming one box means that box. That is how the replay, the endpoint
    #: bench and a run script that pins a machine keep working unchanged with
    #: the pool sitting in the same `.env`.
    pinned = os.environ.get("MEMPA_JUDGE_URL", "").strip()
    if pinned:
        #: Pinning one box is deliberate and has to say so. `.env` is untracked,
        #: so the pool's variables are one careless comment — or one rebuilt
        #: machine — away from disappearing, and without this check the round
        #: would quietly fall back to whatever single URL was lying around and
        #: judge every session on one box with nobody the wiser. The reason is
        #: free text and goes in the preflight line, so a pinned round says on
        #: its own record why it was pinned.
        if not os.environ.get("MEMPA_JUDGE_PIN", "").strip():
            raise JudgeUnavailable(
                "MEMPA_JUDGE_URL pins the judge to one box, which is only ever "
                "deliberate: set MEMPA_JUDGE_PIN to why (a replay, an endpoint "
                "bench, a round that must be attributable to one machine) and "
                "the round starts. If you did not mean to pin anything, the "
                "pool's MEMPA_JUDGE_URL_<n>/_API_KEY_<n>/_WEIGHT_<n> are "
                "missing from .env and need putting back."
            )
        key = (api_key or os.environ.get("MEMPA_JUDGE_API_KEY", "")).strip()
        if not key:
            raise JudgeUnavailable(
                "MEMPA_JUDGE_API_KEY is not set, and the judge endpoint takes no "
                "anonymous calls. Put the key in .env, or run with MEMPA_JUDGE=off."
            )
        #: The reason rides on the endpoint name, so it reaches the judge_call
        #: rows and the log line through the field that already carries which
        #: box answered — a pinned round is legible afterwards without a second
        #: field. Whitespace becomes `_` because that log line is parsed as
        #: space-separated `key=value`.
        reason = "_".join(os.environ.get("MEMPA_JUDGE_PIN", "").split())
        return [
            JudgeEndpoint(
                name=f"pinned:{reason}" if reason else "pinned",
                base_url=normalise_base_url(pinned),
                api_key=key,
            )
        ]
    endpoints: list[JudgeEndpoint] = []
    index = 1
    while True:
        url = os.environ.get(f"MEMPA_JUDGE_URL_{index}", "").strip()
        if not url:
            break
        key = os.environ.get(f"MEMPA_JUDGE_API_KEY_{index}", "").strip()
        if not key:
            raise JudgeUnavailable(
                f"MEMPA_JUDGE_URL_{index} is set but MEMPA_JUDGE_API_KEY_{index} "
                f"is not, and the judge endpoint takes no anonymous calls. The "
                f"two boxes do not share a key."
            )
        endpoints.append(
            JudgeEndpoint(
                name=os.environ.get(f"MEMPA_JUDGE_NAME_{index}", "").strip()
                or f"node{index}",
                base_url=normalise_base_url(url),
                api_key=key,
                weight=_env_float(f"MEMPA_JUDGE_WEIGHT_{index}", 1.0),
                model=os.environ.get(f"MEMPA_JUDGE_MODEL_{index}", "").strip(),
            )
        )
        index += 1
    if endpoints:
        if any(endpoint.weight <= 0 for endpoint in endpoints):
            raise JudgeUnavailable(
                "every MEMPA_JUDGE_WEIGHT_<n> must be positive; a zero share is "
                "a box left out of the pool, so leave it out of the numbering."
            )
        return endpoints
    url = (base_url or os.environ.get("MEMPA_JUDGE_URL", "")).strip()
    if not url:
        raise JudgeUnavailable(
            "MEMPA_JUDGE_URL is not set, and the judge has no default "
            "endpoint. Put the URL in .env, or run with MEMPA_JUDGE=off."
        )
    key = (api_key or os.environ.get("MEMPA_JUDGE_API_KEY", "")).strip()
    if not key:
        raise JudgeUnavailable(
            "MEMPA_JUDGE_API_KEY is not set, and the judge endpoint takes no "
            "anonymous calls. Put the key in .env, or run with MEMPA_JUDGE=off."
        )
    return [
        JudgeEndpoint(name="single", base_url=normalise_base_url(url), api_key=key)
    ]


#: Concurrency is 1 per process, and a correctness choice rather than a
#: performance one: a shared judge took a call from 2 seconds to 89 (§6).
_CALL_LOCK = threading.Lock()

#: Proxy handlers empty: the endpoint must never be sent through the
#: environment's HTTP proxy (see the module docstring).
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _urlopen(request: urllib.request.Request, timeout: float):
    return _OPENER.open(request, timeout=timeout)


def normalise_base_url(url: str) -> str:
    """The OpenAI-compatible base, with `/v1` exactly once."""
    url = url.rstrip("/")
    if url.endswith("/v1"):
        return url
    return f"{url}/v1"


class _Transient(RuntimeError):
    """A failure `_stream` retries in place: no complete judge answer arrived."""


#: The transport failures a WAN endpoint adds to the status codes: a TLS EOF, a
#: reset, a connection dropped with the body half sent. `http.client` raises
#: these from inside the streaming loop, past the point `urllib` would have
#: wrapped them in a `URLError`.
_TRANSPORT_ERRORS = (
    ssl.SSLError,
    http.client.HTTPException,
    ConnectionError,
    EOFError,
)


def _retryable_status(code: int) -> bool:
    """429 and every 5xx. `RETRY_STATUSES` names the two seen most."""
    return code in RETRY_STATUSES or 500 <= code < 600


def _retryable_error_frame(error: Any) -> bool:
    """Only explicit OpenRouter transient codes, never message-text guesses.

    Chat Completions documents numeric HTTP codes and the legacy streaming
    ``server_error`` code. A conflicting typed category stays fatal.
    https://openrouter.ai/docs/api_reference/errors-and-debugging
    https://openrouter.ai/docs/api_reference/streaming
    """
    if not isinstance(error, dict):
        return False
    metadata = error.get("metadata")
    error_type = metadata.get("error_type") if isinstance(metadata, dict) else None
    if error_type is not None and (not isinstance(error_type, str) or error_type not in {
        "server", "rate_limit_exceeded", "provider_overloaded", "provider_unavailable"
    }):
        return False
    code = error.get("code")
    return (type(code) is int and _retryable_status(code)) or code == "server_error"


class JudgeUnavailable(RuntimeError):
    """The judge endpoint is not answering, or the pinned model is not on it.

    Never recovered from by substituting another model: half a judge is worse
    than none, because it records a batch of sessions against another standard.
    """


@dataclass(frozen=True)
class JudgeCall:
    """What one call returned, before any parsing."""

    raw: str
    seconds: float
    prompt_tokens: int | None
    eval_tokens: int | None
    num_predict: int = 0
    done_reason: str = ""
    #: The pool entry that answered, so a verdict can be traced back to the box
    #: that produced it. Empty only where a call was built by hand in a test.
    endpoint: str = ""
    #: `usage.cost` where the endpoint sends it (OpenRouter puts it inline on
    #: the last stream frame), `None` for a box of the lab's own, which costs
    #: nothing per call. The log line prints it, so a round on a paid endpoint
    #: adds up from its own log rather than from the dashboard.
    cost_usd: float | None = None
    #: `usage.completion_tokens_details.reasoning_tokens` where the endpoint
    #: sends it, `None` where it does not. `eval_tokens` counts the reasoning
    #: pass too, so this is what separates the two halves of the generation.
    reasoning_tokens: int | None = None

    @property
    def thinking_cut(self) -> bool:
        """The reasoning pass was stopped by `THINKING_TOKEN_BUDGET` rather than
        by the model finishing it. The replay counts these, so a round can say
        how often the judge was cut off mid-thought."""
        return (
            self.reasoning_tokens is not None
            and self.reasoning_tokens >= THINKING_TOKEN_BUDGET
        )

    @property
    def answer_tokens(self) -> int | None:
        """`eval_tokens` less the reasoning pass — the half `num_predict` sizes."""
        if self.eval_tokens is None:
            return None
        return self.eval_tokens - (self.reasoning_tokens or 0)

    @property
    def truncated(self) -> bool:
        """The generation stopped on the token cap, not on the model finishing.

        `finish_reason` says so; when a frame carries none, a completion count
        that reached the cap says the same thing. A truncated reply is
        well-formed JSON up to the cut, so the parse rejects it and nothing
        downstream can tell the two apart — which is why `guards.ask` widens
        instead of re-asking (§4(1)).
        """
        if self.done_reason:
            return self.done_reason == "length"
        answer = self.answer_tokens
        return self.num_predict > 0 and answer is not None and answer >= self.num_predict


class JudgeClient:
    """One process, one lane, one model."""

    def __init__(
        self,
        *,
        model: str | None = None,
        base_url: str | None = None,
        deadline_s: float | None = None,
    ) -> None:
        #: The most recent call, so the session can put its latency on the
        #: record it writes. The per-call log line is stderr only, and nothing
        #: in a run directory keeps it, so the batch report had no latency to
        #: read (report of 2026-09-09).
        self.last_call: "JudgeCall | None" = None
        self.model = model or os.environ.get("MEMPA_JUDGE_MODEL", "").strip() or DEFAULT_MODEL
        self.endpoints = load_endpoints(base_url=base_url)
        #: What the round is pointed at. In pool mode there is more than one, so
        #: these two name the first entry: the preflight line and the "not
        #: reachable" errors stay readable, and `endpoints` is what the calls
        #: actually go to.
        self.base_url = self.endpoints[0].base_url
        self.api_key = self.endpoints[0].api_key
        self._random = _DRAW
        #: The whole pool in the order this client will use it: the box it is
        #: bound to first, then its fallbacks. Drawn once, here.
        self._order = self._draw_order()
        #: How far down that order the failures have pushed this client. It
        #: moves forward within one pass of the pool: a box that dropped a call
        #: is not drawn again while another one is left to try, because
        #: rebinding mid-session is the thing sticky binding exists to avoid.
        #: The one move backwards is the end of the list coming round to the
        #: head again (`POOL_PASSES`), where there is no box left to prefer.
        self._bound = 0
        self.deadline_s = (
            deadline_s if deadline_s is not None else _env_float(
                "MEMPA_JUDGE_DEADLINE_S", DEFAULT_DEADLINE_S
            )
        )

    # ------------------------------------------------------------- endpoints

    def _draw_order(self) -> list[JudgeEndpoint]:
        """The pool, shuffled by weight: first place is the binding.

        Weighted without replacement, so the first entry is drawn at exactly
        its share of the total weight — the owner's 1/3 LAN, 2/3 workstation —
        and the rest fall in behind it as the fallbacks, in the same weighted
        order.

        `MEMPA_JUDGE_PRIMARY=<n>` replaces the draw with an order: entry `<n>`
        first in every session and every process, the others behind it in index
        order. Nothing else changes — the binding is still first place, and the
        failover still walks down the list and stays where it lands.
        """
        primary = self._primary_index()
        if primary is not None:
            head = self.endpoints[primary]
            return [head] + [e for e in self.endpoints if e is not head]
        order: list[JudgeEndpoint] = []
        pool = list(self.endpoints)
        while pool:
            draw = self._random.uniform(0, sum(e.weight for e in pool))
            for endpoint in pool:
                draw -= endpoint.weight
                if draw <= 0:
                    break
            order.append(endpoint)
            pool.remove(endpoint)
        return order

    def _primary_index(self) -> int | None:
        """`MEMPA_JUDGE_PRIMARY` as a 0-based index into `endpoints`, or None.

        Out of range is an error rather than a silent fall back to the draw: a
        round that meant to pin its primary and got the dice instead is exactly
        the state the pool's own missing-variables check exists to stop.
        """
        raw = os.environ.get("MEMPA_JUDGE_PRIMARY", "").strip()
        if not raw:
            return None
        try:
            index = int(raw)
        except ValueError as exc:
            raise JudgeUnavailable(
                f"MEMPA_JUDGE_PRIMARY must be the number of a pool entry "
                f"(1..{len(self.endpoints)}), got {raw!r}."
            ) from exc
        if not 1 <= index <= len(self.endpoints):
            raise JudgeUnavailable(
                f"MEMPA_JUDGE_PRIMARY={index} names no pool entry; the pool has "
                f"{len(self.endpoints)} ({', '.join(e.name for e in self.endpoints)})."
            )
        return index - 1

    @property
    def endpoint(self) -> JudgeEndpoint:
        """The box this client is bound to right now."""
        return self._order[self._bound]

    @property
    def preflight_endpoints(self) -> tuple[JudgeEndpoint, ...]:
        """Check the current preference first without rebinding the client."""
        return tuple(self._order[self._bound:] + self._order[:self._bound])

    def model_for(self, endpoint: JudgeEndpoint) -> str:
        """The model id this box serves: its own where it has one."""
        return endpoint.model or self.model

    # ------------------------------------------------------------------ call

    def generate(
        self,
        *,
        system: str,
        prompt: str,
        allowed: Mapping[str, Sequence[str]],
        num_predict_kind: str = "yes_no",
        num_predict: int | None = None,
        endpoint: JudgeEndpoint | None = None,
    ) -> JudgeCall:
        """`num_predict` overrides the sizing; `guards.ask` sends it to widen
        one call that came back cut, and nothing else passes it. `endpoint`
        pins one box instead of drawing from the pool; the preflight passes it
        to check every box, and nothing else does."""
        tokens = (
            num_predict
            if num_predict is not None
            else effective_num_predict(num_predict_kind, len(allowed))
        )
        payload = {
            # Replaced per attempt with `model_for(served)`: the fallback box
            # serves another model id, and the payload follows the box.
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
            # The tiers size the answer; the thinking budget sits on top of
            # them, so a long reasoning pass can never eat the verdict's room.
            "max_tokens": tokens + THINKING_TOKEN_BUDGET,
            "thinking_token_budget": THINKING_TOKEN_BUDGET,
            "stream": True,
            "stream_options": {"include_usage": True},
            "chat_template_kwargs": CHAT_TEMPLATE_KWARGS,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "verdicts", "schema": verdict_schema(allowed)},
            },
        }
        question_ids = list(allowed)
        deadline = effective_deadline_s(len(allowed), floor=self.deadline_s)
        pinned = endpoint is not None
        failed_over = False
        #: How many times this call has walked the pool (`POOL_PASSES`).
        walked = 1
        with _CALL_LOCK:
            # The call's clock starts inside the lock, never before it: the Test
            # line runs judge calls from worker threads, and queueing behind
            # another thread's call is not time this box has spent. Started
            # outside, the wait ate the deadline and a healthy box reported a
            # timeout it never caused. It measures the call for the record only:
            # the deadline is per attempt and `_stream` keeps that clock.
            started = time.monotonic()
            while True:
                served = endpoint if pinned else self.endpoint
                payload["model"] = self.model_for(served)
                try:
                    chunks, final = self._stream(
                        payload, question_ids, deadline, served
                    )
                    break
                except JudgeTimeout as exc:
                    # The box dropped this call mid-stream. Owner ruling
                    # 2026-09-14: try it once on the other box before the
                    # session is voided — a dropped socket is the box's
                    # failure, not the turn's outcome, and the rerun pinned to
                    # the other box is what a void buys anyway. Once only,
                    # never back onto the box that just dropped it, and not
                    # once one attempt has run past its failover window. The
                    # in-place retries of `_stream` come first: by here the box
                    # has dropped the call on every one of them.
                    if (
                        pinned
                        or failed_over
                        or self._bound + 1 == len(self._order)
                        or exc.seconds >= FAILOVER_MAX_DEADLINES * deadline
                    ):
                        raise
                    failed_over = True
                    self._bound += 1
                    print(
                        f"judge_failover from={served.name} "
                        f"from_model={self.model_for(served)} "
                        f"to={self.endpoint.name} "
                        f"to_model={self.model_for(self.endpoint)} "
                        f"reason=timeout call={','.join(question_ids)}",
                        file=sys.stderr,
                        flush=True,
                    )
                except JudgeUnavailable as exc:
                    # The bound box is down, refusing, or has spent its retries:
                    # move this client to the next one and let it take the call.
                    # At the end of the list the walk comes back round to the
                    # head (`POOL_PASSES`) rather than ending the round, because
                    # a pool whose fallbacks are all unreachable is exactly when
                    # the primary is the only box left worth asking.
                    end_of_pool = self._bound + 1 == len(self._order)
                    if pinned or (
                        end_of_pool
                        and (len(self._order) == 1 or walked >= POOL_PASSES)
                    ):
                        raise
                    if end_of_pool:
                        walked += 1
                        self._bound = 0
                    else:
                        self._bound += 1
                    print(
                        f"judge_failover from={served.name} "
                        f"from_model={self.model_for(served)} "
                        f"to={self.endpoint.name} "
                        f"to_model={self.model_for(self.endpoint)} "
                        f"reason=unavailable call={','.join(question_ids)} "
                        f"detail={'_'.join(str(exc).split())[:160]}",
                        file=sys.stderr,
                        flush=True,
                    )
        usage = final.get("usage") or {}
        details = usage.get("completion_tokens_details") or {}
        call = JudgeCall(
            raw="".join(chunks),
            seconds=time.monotonic() - started,
            prompt_tokens=usage.get("prompt_tokens"),
            eval_tokens=usage.get("completion_tokens"),
            num_predict=tokens,
            done_reason=str(final.get("finish_reason") or ""),
            reasoning_tokens=details.get("reasoning_tokens"),
            cost_usd=usage.get("cost"),
            endpoint=served.name,
        )
        self.last_call = call
        _log_judge_call(self.model_for(served), call, question_ids)
        return call

    def _stream(
        self,
        payload: dict[str, Any],
        question_ids: Sequence[str],
        deadline_s: float,
        endpoint: JudgeEndpoint,
    ) -> tuple[list[str], dict[str, Any]]:
        """The SSE frames, concatenated; bounded retry on a transport failure.

        Retried in place, on the same box: 429 and any 5xx, a refused or reset
        connection, a TLS EOF mid-stream, a 200 whose body carried no content at
        all, and — since the owner's ruling of 2026-09-18 — a call the box
        dropped on its deadline. Explicit transient SSE errors follow the same
        budget. No complete answer arrived, so this does not request a second
        verdict. A reply that arrived and was merely bad is not retried here and
        never has been; `guards.ask` owns that path.

        A timeout was the one failure with no retry at all, on the grounds that
        the box had spent the deadline: that cost a live round, because one slow
        OpenRouter call meant the primary was abandoned after zero retries and
        the run died on two unreachable fallbacks. So each attempt now carries
        its own whole deadline (`deadline_s` bounds an attempt, not the call),
        and a timeout is retried in place like the rest — with no backoff, since
        the attempt has already waited a deadline out on this box. The one
        timeout still not retried is the runaway: an attempt that ran through
        `FAILOVER_MAX_DEADLINES` of its own deadline is handed straight back to
        `generate`, whose failover is what that state has.

        When the retries run out the endpoint is unavailable (a timeout raises
        its `JudgeTimeout` instead, so the session-void path still reads it as
        one), and `generate` moves the client to the fallback.
        """
        attempts = max(0, int(_env_float("MEMPA_JUDGE_RETRIES", DEFAULT_RETRIES)))
        retried = 0
        while True:
            # Every attempt's own clock: the deadline it is measured against is
            # a whole one, not what the attempts before it left over.
            started = time.monotonic()
            pause = RETRY_PAUSE_S * (2 ** retried)
            try:
                chunks, final = self._stream_once(
                    payload, question_ids, started, deadline_s, endpoint
                )
                if not chunks:
                    raise _Transient(
                        f"the judge endpoint {endpoint.name} at "
                        f"{endpoint.base_url} answered with an empty body for "
                        f"model {payload.get('model')!r}"
                    )
                return chunks, final
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", "replace")[:400]
                reason = (
                    f"the judge endpoint {endpoint.name} at {endpoint.base_url} "
                    f"returned HTTP {exc.code} for model "
                    f"{payload.get('model')!r}: {body}"
                )
                if not _retryable_status(exc.code):
                    raise JudgeUnavailable(reason) from exc
                cause: BaseException = exc
            except urllib.error.URLError as exc:
                reason = (
                    f"the judge endpoint {endpoint.name} is not reachable at "
                    f"{endpoint.base_url}: {exc.reason}"
                )
                cause = exc
            except _Transient as exc:
                reason = str(exc)
                cause = exc
            except (JudgeTimeout, socket.timeout, TimeoutError) as exc:
                timeout = (
                    exc
                    if isinstance(exc, JudgeTimeout)
                    else JudgeTimeout(
                        question_ids=question_ids,
                        seconds=time.monotonic() - started,
                        partial_response="",
                        partial_tokens=0,
                    )
                )
                if (
                    retried >= attempts
                    or timeout.seconds >= FAILOVER_MAX_DEADLINES * deadline_s
                ):
                    if timeout is exc:
                        raise
                    raise timeout from exc
                reason = (
                    f"the judge endpoint {endpoint.name} at {endpoint.base_url} "
                    f"dropped the call on its deadline after "
                    f"{timeout.seconds:.0f}s for model {payload.get('model')!r}"
                )
                cause = exc
                # No backoff: this attempt already waited a whole deadline out.
                pause = 0.0
            except _TRANSPORT_ERRORS as exc:
                # A TLS EOF or a reset mid-stream: the connection died with the
                # body half sent, which over the WAN is the common failure and
                # is not an answer.
                reason = (
                    f"the judge endpoint {endpoint.name} at {endpoint.base_url} "
                    f"dropped the connection: {exc!r}"
                )
                cause = exc
            if retried >= attempts:
                raise JudgeUnavailable(
                    f"{reason} (gave up after {retried} "
                    f"{'retry' if retried == 1 else 'retries'})"
                ) from cause
            retried += 1
            print(
                f"judge_retry endpoint={endpoint.name} "
                f"model={payload.get('model')} attempt={retried}/{attempts} "
                f"pause_s={pause:.0f} call={','.join(question_ids)} "
                f"reason={'_'.join(reason.split())[:160]}",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(pause)

    def _stream_once(
        self,
        payload: dict[str, Any],
        question_ids: Sequence[str],
        started: float,
        deadline_s: float,
        endpoint: JudgeEndpoint,
    ) -> tuple[list[str], dict[str, Any]]:
        request = urllib.request.Request(
            f"{endpoint.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {endpoint.api_key}",
            },
        )
        chunks: list[str] = []
        final: dict[str, Any] = {}
        try:
            with _urlopen(request, timeout=deadline_s) as response:
                for line in response:
                    elapsed = time.monotonic() - started
                    if elapsed > deadline_s:
                        raise JudgeTimeout(
                            question_ids=question_ids,
                            seconds=elapsed,
                            partial_response="".join(chunks),
                            partial_tokens=len(chunks),
                        )
                    line = line.strip()
                    if not line.startswith(b"data:"):
                        continue
                    data = line[len(b"data:") :].strip()
                    if data == b"[DONE]":
                        break
                    frame = json.loads(data)
                    # OpenRouter also sends errors alongside a terminal choice
                    # with finish_reason=error after HTTP 200 is committed.
                    if frame.get("error"):
                        error = frame["error"]
                        message = (
                            error.get("message")
                            if isinstance(error, dict)
                            else error
                        )
                        if not isinstance(message, str) or not message:
                            message = "unspecified provider error"
                        code = error.get("code") if isinstance(error, dict) else None
                        code_label = (
                            str(code) if type(code) is int and 100 <= code < 600
                            else "server_error" if code == "server_error" else "unknown"
                        )
                        reason = (
                            f"the judge endpoint {endpoint.name} at "
                            f"{endpoint.base_url} returned an error frame for "
                            f"model {self.model!r} (code={code_label}): {message}"
                        )
                        if not final.get("finish_reason") and _retryable_error_frame(error):
                            raise _Transient(reason)
                        raise JudgeUnavailable(reason)
                    for choice in frame.get("choices") or []:
                        content = (choice.get("delta") or {}).get("content")
                        if content:
                            chunks.append(content)
                        if choice.get("finish_reason"):
                            final["finish_reason"] = choice["finish_reason"]
                    if frame.get("usage"):
                        final["usage"] = frame["usage"]
        except (socket.timeout, TimeoutError) as exc:
            raise JudgeTimeout(
                question_ids=question_ids,
                seconds=time.monotonic() - started,
                partial_response="".join(chunks),
                partial_tokens=len(chunks),
            ) from exc
        return chunks, final

    # ----------------------------------------------------------------- probe

    def probe(self, endpoint: JudgeEndpoint | None = None) -> JudgeCall:
        """One real call with a real-shaped payload, before any run.

        Phase 2 uses it as the run gate: it must come back parseable against the
        §3 schema, or the round does not start.
        """
        from simulator.judge.catalogue import PROBE_PROMPT, PROBE_SYSTEM
        from simulator.judge.guards import parse_verdicts

        allowed = {"carries_an_ask": ASK_GATE_VALUES}
        call = self.generate(
            system=PROBE_SYSTEM,
            prompt=PROBE_PROMPT,
            allowed=allowed,
            num_predict_kind="yes_no",
            endpoint=endpoint,
        )
        parse_verdicts(call.raw, allowed=allowed)
        return call


def _log_judge_call(
    model: str, call: JudgeCall, question_ids: Sequence[str]
) -> None:
    """One unbuffered stderr line per call, printed the moment it returns.

    Same shape as `simulator.providers.openai_compat_provider._log_openrouter_call`.
    `cost_usd` is what the endpoint reported, and 0 for a box of the lab's own,
    which reports none.
    """
    line = (
        f"judge_call model={model} endpoint={call.endpoint} "
        f"seconds={call.seconds:.1f} "
        f"prompt_tokens={call.prompt_tokens} completion_tokens={call.eval_tokens} "
        f"num_predict={call.num_predict} reasoning_tokens={call.reasoning_tokens} "
        f"cost_usd={call.cost_usd or 0} questions={','.join(question_ids)} "
        f"verdicts={_verdict_summary(call.raw)}"
    )
    print(line, file=sys.stderr, flush=True)


def _verdict_summary(raw: str) -> str:
    """Best effort, for the log line only — never a parse the guards trust."""
    try:
        payload = json.loads(strip_fences(raw))
        items = payload["verdicts"]
        return ",".join(f"{item['question_id']}={item['verdict']}" for item in items)
    except Exception:
        return "unparsed"


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc
