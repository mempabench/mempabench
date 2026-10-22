"""One judge per simulator session: the client, the budget, and the void door.

Phase 1 built the module; this is the piece the ladder holds. It owns three
things the call sites should not each re-decide:

- **the mode.** `MEMPA_JUDGE` is `off` (default — nothing changes for a run or
  a test that says nothing) or `live` (the judge decides; a rule it decides is
  simply not checked when it is off). There is no per-rule switch and no
  per-variant flag: contract §13 (ii) makes the prompt file the only patch
  channel.
- **the budget.** `MEMPA_JUDGE_TURN_BUDGET` question groups per candidate
  generation, reset by `begin_generation` (§4(3)). The ask gate spends against
  the generation that produced the candidate it is judging, which is the last
  one begun.
- **the void door.** A failure exception lands on the transcript as its own
  event and raises `SessionInvalidError` with that event name as the rule (§4,
  §13 i). There is no fallback model, no degrade to the regex, and no "played
  with a diagnostic" path: the simulator is an instrument, and an instrument
  that cannot read gives no number.
- **the advisory door.** A rule whose prompt header says `status: unmeasured`
  is asked and recorded like any other, and then decides nothing: its answer is
  emitted as `judge_advisory` and the rule returns its non-firing value. The
  header is the rule's own property, so promoting one means replaying it and
  re-stamping that file — no per-rule live list, no flag.
- **the release door**, for the one failure that is not the instrument's
  (owner ruling 2026-09-09, 用C). A reference that will not resolve twice — the
  first ask and the one feedback re-ask — is an answer that was never
  substantiated, not an infrastructure fault: the event is recorded, the group
  returns no verdict, and the turn goes through. A timeout, an unparseable
  reply and a dead model still void.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any, Callable, NoReturn

from simulator.judge.catalogue import (
    BY_RULE,
    RULE_PROMPTS,
    QuestionGroup,
    RuleSpec,
    TurnContext,
    render,
)
from simulator.judge.client import JudgeClient, JudgeUnavailable, _urlopen
from simulator.judge.guards import (
    JudgeEvidenceInvalid,
    JudgeFailure,
    TurnBudget,
    Verdict,
    ask as guarded_ask,
)
from simulator.output import SessionInvalidError

#: `off` first, so an unset environment is the behaviour every existing run and
#: test already has.
JUDGE_MODES = ("off", "live")

#: The per-rule catalogue, by name: a call site names a rule and asks for it.
SPECS: dict[str, RuleSpec] = dict(BY_RULE)

#: What a released group's rule answers: the value that fires nothing. Every
#: compose but one returns a boolean, so `False` is the default; the one that
#: does not is written down here, and so is the ask gate, whose non-firing value is
#: the permissive one — a turn nobody could substantiate an objection to is
#: played, not thrown away (owner ruling 2026-09-09).
RELEASED_VALUE: dict[str, Any] = {
    "carries_an_ask": True,
    "probe_fact_missing": [],
}


def current_mode() -> str:
    raw = os.environ.get("MEMPA_JUDGE", "").strip().lower() or "off"
    if raw not in JUDGE_MODES:
        raise ValueError(
            f"MEMPA_JUDGE must be one of {list(JUDGE_MODES)}, got {raw!r}"
        )
    return raw


class JudgeSession:
    """The judge for one simulator session."""

    def __init__(
        self,
        *,
        mode: str = "off",
        client: Any | None = None,
        on_event: Callable[..., Any] | None = None,
        session_id: str = "",
    ) -> None:
        if mode not in JUDGE_MODES:
            raise ValueError(f"mode must be one of {list(JUDGE_MODES)}, got {mode!r}")
        self.mode = mode
        self.session_id = session_id
        self.on_event = on_event
        self.client = client
        if self.client is None and mode != "off":
            self.client = JudgeClient()
        self._budget = TurnBudget(session_id or "session")
        self._beat = ""
        #: Whether the last `ask` was released rather than answered, so `judge`
        #: knows the empty dict is a release and not an empty question group.
        self._released = False
        #: The verdicts of the most recent call, so a call site that has to put
        #: the judge's own words on a record (the ask gate's rejected-attempt
        #: row) does not have to thread them back out of `compose`.
        self.last_verdicts: dict[str, Verdict] = {}

    @classmethod
    def from_env(
        cls,
        *,
        session_id: str = "",
        on_event: Callable[..., Any] | None = None,
        client: Any | None = None,
    ) -> "JudgeSession":
        return cls(
            mode=current_mode(),
            client=client,
            on_event=on_event,
            session_id=session_id,
        )

    # ------------------------------------------------------------------ mode

    @property
    def enabled(self) -> bool:
        """The judge is called at all."""
        return self.mode != "off"

    # ---------------------------------------------------------------- budget

    def begin_generation(self, turn_uid: str, *, beat: str = "") -> None:
        """A new candidate generation: the question-group counter starts over."""
        self._budget = TurnBudget(turn_uid)
        if beat:
            self._beat = beat

    # ------------------------------------------------------------------- ask

    def ask(self, group: QuestionGroup) -> dict[str, Verdict]:
        """One question group, or nothing at all when the group is empty.

        A group that answered writes one `judge_call` record: which rule, which
        generation, how many groups that generation had spent by then, the
        client's own latency for the call, and every verdict. The batch report
        (`scripts/diagnostics/regression_counters.py`) is read off these — hits
        per rule, calls per session, the worst generation, latency p95 — and
        nothing in a run directory kept the stderr log line it used to need.

        `verdicts` answers *what* the judge said; `answers` answers *why*. It
        carries, in question order and untruncated, each verdict with the
        evidence it quoted and the reason it gave — the sentence a `violated`
        pointed at, which nothing kept once the candidate turn went away, and
        without which a voided session cannot be read back afterwards.
        """
        self._released = False
        if group.is_empty:
            self.last_verdicts = {}
            return {}
        system, prompt = render(group)
        try:
            verdicts = guarded_ask(
                self.client,
                system=system,
                prompt=prompt,
                allowed=group.allowed(),
                blocks_by_name=group.blocks,
                quotes_on=group.quotes_on(),
                question_blocks=group.question_blocks(),
                num_predict_kind=group.num_predict_kind,
                budget=self._budget,
            )
        except JudgeEvidenceInvalid as exc:
            return self._release(exc)
        except JudgeFailure as exc:
            self._void(exc)
        except JudgeUnavailable as exc:
            self._unavailable(exc)
        self.last_verdicts = {verdict.question_id: verdict for verdict in verdicts}
        self.emit(
            "judge_call",
            rule=group.rule,
            turn_uid=self._budget.turn_uid,
            group=self._budget.used,
            seconds=getattr(getattr(self.client, "last_call", None), "seconds", None),
            # Which box answered. With the pool the verdicts in one run come
            # from whichever machine that session drew, and a disagreement is
            # only readable after the fact if the record says which one.
            endpoint=getattr(getattr(self.client, "last_call", None), "endpoint", None),
            verdicts={
                verdict.question_id: verdict.verdict for verdict in verdicts
            },
            answers=[
                {
                    "question_id": question_id,
                    "verdict": self.last_verdicts[question_id].verdict,
                    "evidence": self.last_verdicts[question_id].evidence,
                    "reason": self.last_verdicts[question_id].reason,
                }
                for question_id in group.allowed()
            ],
        )
        return dict(self.last_verdicts)

    def last_evidence(self, question_id: str) -> str:
        verdict = self.last_verdicts.get(question_id)
        return verdict.evidence if verdict else ""

    def judge(self, rule: str, ctx: TurnContext) -> Any:
        """Build this rule's call, ask it, and compose its answer in code.

        A released call composes nothing: there are no answers to compose, and
        `compose` reading an empty dict would return the FIRING value for the
        ask gate — the one rule whose absent verdict must let the turn through.
        `RELEASED_VALUE` says what each rule answers instead.
        """
        spec = SPECS[rule]
        group = spec.build(ctx)
        verdicts = self.ask(group)
        if self._released:
            return RELEASED_VALUE.get(rule, False)
        composed = spec.compose(verdicts, ctx)
        if RULE_PROMPTS[rule].status == "unmeasured":
            # Asked, recorded, and not allowed to decide: no replay has scored
            # this text, so it may not block or release a live turn. The only
            # way up is to replay it and re-stamp the prompt file's header —
            # there is no list and no flag (contract §13 ii).
            self.emit("judge_advisory", rule=rule, composed=composed)
            return RELEASED_VALUE.get(rule, False)
        return composed

    def _release(self, exc: JudgeEvidenceInvalid) -> dict[str, Verdict]:
        """The answer was never substantiated: record it and let the turn go.

        Same event as the void door wrote, plus `released`, so a read can count
        how often the judge could not point at what it claimed. The session is
        not void — twice unable to give a valid reference is the judge failing
        to prove its case, not the instrument failing to read (owner ruling
        2026-09-09, "judge 判两次，都拦不住的话，直接放行").
        """
        event = dict(exc.event)
        kind = str(event.pop("event"))
        self.emit(kind, released=True, **event)
        self._released = True
        self.last_verdicts = {}
        return {}

    def _void(self, exc: JudgeFailure) -> NoReturn:
        event = dict(exc.event)
        kind = str(event.pop("event"))
        self.emit(kind, **event)
        raise SessionInvalidError(
            kind,
            beat=self._beat,
            attempts=int(event.get("attempts") or 1),
        ) from exc

    def _unavailable(self, exc: JudgeUnavailable) -> NoReturn:
        """The endpoint went away, or the model was unloaded, after the
        preflight.

        The session is void the way the four failure events are — the record
        says why it died — but this one does not stop at the session. The rest
        of the batch would run with no judge at all, recorded against another
        standard, so the `SessionInvalidError` carries the original as its
        cause and `SimulatorLoop.run_session` re-raises that instead of
        letting the harness move to the next session. The owner's words for
        this state: 模型不能用，这一轮跑不了.
        """
        self.emit(
            "judge_unavailable",
            reason=str(exc),
            model=getattr(self.client, "model", ""),
        )
        raise SessionInvalidError(
            "judge_unavailable", beat=self._beat, attempts=1
        ) from exc

    # --------------------------------------------------------------- records

    def emit(self, kind: str, **payload: Any) -> None:
        if self.on_event is not None:
            self.on_event(kind, **payload)

    def log(self, line: str) -> None:
        """Unbuffered, the moment it happens — the client logs each call, this
        logs each decision."""
        print(line, file=sys.stderr, flush=True)


# ------------------------------------------------------------------ preflight


def preflight(mode: str | None = None, *, client: Any | None = None) -> None:
    """Require a configured endpoint serving its exact model and valid schema.

    Check the current preference first, then the declared fallbacks. At least
    one must pass both the model listing and a pinned probe. This does not
    change endpoint preference or the clients created for later sessions.
    A single endpoint retains the same strict refusal behavior.
    """
    mode = current_mode() if mode is None else mode
    if mode == "off":
        return
    client = client or JudgeClient()
    endpoints = getattr(client, "endpoints", None)
    verified_endpoint = None
    if endpoints:
        bound_endpoint = getattr(client, "endpoint", None)
        order = getattr(client, "preflight_endpoints", None)
        if order is None:
            order = ([bound_endpoint] if bound_endpoint is not None else []) + [
                endpoint for endpoint in endpoints if endpoint is not bound_endpoint
            ]
        failures = []
        for endpoint in order:
            try:
                _check_endpoint(mode, client, endpoint)
                if verified_endpoint is None:
                    verified_endpoint = endpoint
            except Exception as exc:
                failures.append(exc)
                warning = "primary_down" if endpoint is bound_endpoint else "fallback_down"
                print(
                    f"judge_preflight mode={mode} "
                    f"model={_endpoint_model(client, endpoint)} "
                    f"endpoint={endpoint.name} url={endpoint.base_url} "
                    f"WARNING {warning} "
                    f"detail={'_'.join(str(exc).split())[:200]}",
                    file=sys.stderr,
                    flush=True,
                )
        if verified_endpoint is None:
            raise failures[0]
    else:
        available = _installed_models(client.base_url)
        _refuse_unless_served(client.model, client.base_url, available)
        call = client.probe()
        _print_preflight(mode, client.model, "", client.base_url, call)
    bound = getattr(client, "endpoint", None)
    if bound is not None:
        checked = verified_endpoint if endpoints else bound
        label = (
            f"bound={bound.name}" if checked is bound else
            f"verified_endpoint={checked.name} configured_bound={bound.name}"
        )
        print(
            f"judge_preflight mode={mode} model={_endpoint_model(client, checked)} "
            f"{label} url={checked.base_url} ok",
            file=sys.stderr,
            flush=True,
        )


def _endpoint_model(client: Any, endpoint: Any) -> str:
    """The model id this box serves — its own where the pool gave it one."""
    model_for = getattr(client, "model_for", None)
    if model_for is not None:
        return model_for(endpoint)
    return getattr(client, "model", "")


def _check_endpoint(mode: str, client: Any, endpoint: Any) -> None:
    model = _endpoint_model(client, endpoint)
    available = _installed_models(endpoint.base_url, endpoint.api_key)
    _refuse_unless_served(model, endpoint.base_url, available)
    call = client.probe(endpoint)
    _print_preflight(mode, model, endpoint.name, endpoint.base_url, call)


def _refuse_unless_served(model: str, base_url: str, available: set[str]) -> None:
    if model in available:
        return
    #: The listing is truncated because OpenRouter's `/v1/models` is hundreds
    #: of ids long and the whole set in an exception is unreadable.
    listed = sorted(available)
    shown = ", ".join(listed[:12]) + (f", ... ({len(listed)} total)" if len(listed) > 12 else "")
    raise JudgeUnavailable(
        f"judge model {model!r} is not served at {base_url}; the endpoint "
        f"lists {shown}. The pool's fallback is another box with its own "
        f"MEMPA_JUDGE_MODEL_<n>, never another model on this one: point "
        f"MEMPA_JUDGE_URL_<n>/MEMPA_JUDGE_MODEL_<n> at an endpoint serving "
        f"this model, or run with MEMPA_JUDGE=off."
    )


def _print_preflight(
    mode: str, model: str, name: str, base_url: str, call: Any
) -> None:
    print(
        f"judge_preflight mode={mode} model={model} endpoint={name} "
        f"url={base_url} seconds={call.seconds:.1f} ok",
        file=sys.stderr,
        flush=True,
    )


def _installed_models(base_url: str, api_key: str | None = None) -> set[str]:
    key = (api_key or os.environ.get("MEMPA_JUDGE_API_KEY", "")).strip()
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/models",
        headers={"Authorization": f"Bearer {key}"},
    )
    try:
        with _urlopen(request, timeout=10) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise JudgeUnavailable(
            f"the judge endpoint is not reachable at {base_url}: {exc}"
        ) from exc
    return {str(model.get("id") or "") for model in payload.get("data") or []}
