"""The one routine a forced-event tool call goes through, live or offline.

A forced event is replayed twice by two different callers, and until now each
carried its own copy of the same four decisions:

* `harness/forced_event_executor.py` replays it against the run's real stage,
  for the benchmark;
* `replay_world_results()` in `scripts/generate_forced_event_transcript.py`
  replays it against a scratch copy of the persona world, to freeze each read's
  `expected_contains` and to re-check the transcript at validation time.

Two copies of "which argument value is this", "does the gate hold this call",
"does this message open the gate", and "does the world's answer satisfy this
read's `expected_contains`" are two chances to disagree, and the offline one
decides what the live one is later asserted against. So the four decisions live
here once, and both callers run the same code over their own store: the
executor against the stage, the replay against the scratch world.

The state each caller keeps differs -- the executor's gate marker is a file the
state server also reads, the offline replay's is a flag in memory -- so the
marker is injected as a pair of callables rather than assumed to be either.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import yaml

from harness.tool_refs import (
    TASK_FACTS_PRODUCER,
    ToolRefError,
    refs_in,
    resolve_refs,
    resolve_task_fact_refs,
)

#: Runtime tool names are the catalog names behind the MCP server's prefix.
STATE_TOOL_NAME_PREFIX = "mcp_state_"


class ForcedEventReplayError(ValueError):
    """A forced-event item cannot be replayed as authored.

    A `ValueError`, because the offline replay's callers already treat a
    rejected call that way; the executor re-raises it as a
    `ForcedEventExecutionError`.
    """


#: Sizes the world derives from a document's own text. A pinned `content` is
#: checked line by line, and the bytes of a grown document never match the
#: bytes of the pristine one it was frozen from.
_DERIVED_SIZE_FIELDS = frozenset({"bytes"})


def _lines_in_order(actual: str, expected: str) -> bool:
    """Whether every non-empty expected line occurs in `actual`, in order."""
    wanted = [line.strip() for line in expected.splitlines() if line.strip()]
    have = iter(line.strip() for line in actual.splitlines() if line.strip())
    return all(any(line == candidate for candidate in have) for line in wanted)


def result_contains_expected(actual: Any, expected: Any) -> bool:
    """Whether the world's answer satisfies one read's `expected_contains`.

    The fourth decision both replays make, and the one they disagreed on
    longest: the offline replay froze the pins and re-checked them at
    validation time with its own copy of this walk
    (`scripts/generate_forced_event_transcript.py`), the live one checked the
    same pins against the run's real stage with another
    (`harness/forced_event_executor.py`).

    An expected list that is empty asserts that the world returned no members
    (owner ruling 2026-09-12 on Codex F4). It is the writer's own assertion --
    the generator prompt offers "that the result is empty", and a set is the
    one answer `_pin_subset` cannot freeze for them -- so on the vacuous
    reading (`all(... for item in [])`) it claimed nothing, and an event that
    turned on a read finding no candidate went on passing once the world grew
    one. A non-empty expected list still asks only for members.

    A pinned document `content` is a containment claim, not a snapshot. The
    pins are frozen against the pristine fixture, but by the slot the forced
    event replays the accumulation sessions have appended to the same
    documents (owner ruling on state-change semantics: accumulation is
    accepted), so whole-string equality failed a world that still holds every
    fact the pinned PA turn relies on. `content` is satisfied when every
    non-empty pinned line occurs in the actual text in order; appended or
    interleaved lines are allowed, a missing, altered or reordered pinned line
    is not. `bytes` is a size derived from that same text, so it says nothing
    the line walk has not already decided and is not compared when `content`
    is pinned. Every other scalar stays exact.
    """
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return False
        if isinstance(expected.get("content"), str):
            expected = {
                key: value
                for key, value in expected.items()
                if key not in _DERIVED_SIZE_FIELDS
            }
            if not isinstance(actual.get("content"), str):
                return False
            if not _lines_in_order(actual["content"], expected["content"]):
                return False
            expected = {k: v for k, v in expected.items() if k != "content"}
        return all(
            key in actual and result_contains_expected(actual[key], value)
            for key, value in expected.items()
        )
    if isinstance(expected, list):
        if not isinstance(actual, list):
            return False
        if not expected:
            return not actual
        return all(
            any(result_contains_expected(candidate, item) for candidate in actual)
            for item in expected
        )
    return actual == expected


def _in_memory_marker() -> tuple[Callable[[], bool], Callable[[], None]]:
    state = {"open": False}
    return (lambda: state["open"], lambda: state.__setitem__("open", True))


_CATALOG_PATH = Path(__file__).resolve().parents[1] / "data/state_tools/catalog.yaml"


@lru_cache(maxsize=1)
def _catalog_parameter_types() -> dict[str, dict[str, str]]:
    """`{tool: {parameter: declared JSON type}}`, read from the state catalog."""
    catalog = yaml.safe_load(_CATALOG_PATH.read_text(encoding="utf-8")) or {}
    return {
        name: {
            parameter: str((spec or {}).get("type"))
            for parameter, spec in ((definition or {}).get("parameters") or {}).items()
        }
        for name, definition in (catalog.get("tools") or {}).items()
    }


def _coerced(value: Any, declared: str) -> Any:
    """One resolved argument, in the JSON type the catalog declares for it.

    A transcript's `task_facts` are text by contract -- the generator writes
    them as `str(value)` and `validate_gated_values_match_draft` compares the
    same stringification, because they are what the user *says* and
    `harness/gate.py` matches gate key terms against that text. A tool
    parameter, meanwhile, has a declared JSON type. Where an argument is fed
    from a task fact the two contracts meet, and before this they disagreed
    silently: `$task_facts.desk_lamp_amount` resolved to `'38.5'` and reached
    `state_server/store.py` as a `str`, which raised `TypeError: '<=' not
    supported between instances of 'str' and 'int'` from inside the store's own
    arithmetic -- offline in the generator's replay and live in the executor
    alike, since both go through `resolve_arguments`. user_d `evolv_02` is the
    first forced event in any persona to take a non-string catalog type from a
    task fact, which is why nothing caught it until 2026-09-15.

    Only strings are converted, and only into the type the catalog names. Text
    that will not convert is returned unchanged rather than mangled, so the
    generator's `forced_transcript_argument_type` still reports it by name and
    the store still refuses it live. `array` and `object` are left alone. The
    gate is not involved: it keeps comparing the string form.
    """
    if declared == "string" or not isinstance(value, str):
        return value
    text = value.strip()
    if declared == "number":
        # `str()` round-trips the draft's own type: '3' was an int, '38.5' a
        # float, and the catalog's `number` accepts either.
        try:
            return int(text)
        except ValueError:
            pass
        try:
            return float(text)
        except ValueError:
            return value
    if declared == "integer":
        try:
            return int(text)
        except ValueError:
            return value
    if declared == "boolean":
        if text.casefold() in {"true", "false"}:
            return text.casefold() == "true"
    return value


@dataclass
class GatedReplay:
    """Gate state, reference results, and the decisions both replays make."""

    gate: dict[str, Any] | None
    task_facts: dict[str, Any] = field(default_factory=dict)
    #: Whether the gate marker is down, and how to take it down. The executor
    #: passes the marker file's `exists` / `touch`; the offline replay takes
    #: the default in-memory pair.
    gate_is_open: Callable[[], bool] | None = None
    open_gate: Callable[[], None] | None = None
    #: Producer id -> the result it returned. `$call_id.field` reads this.
    results: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.gate_is_open is None or self.open_gate is None:
            is_open, opener = _in_memory_marker()
            self.gate_is_open = self.gate_is_open or is_open
            self.open_gate = self.open_gate or opener

    @property
    def gate_tool_name(self) -> str:
        if not self.gate:
            return ""
        return f"{STATE_TOOL_NAME_PREFIX}{self.gate.get('tool')}"

    def resolve_arguments(
        self,
        *,
        call_id: str,
        arguments: dict[str, Any],
        tool: str | None = None,
    ) -> dict[str, Any]:
        """Every `$…` argument of one authored call, resolved.

        Two producers, in this order. `$task_facts.<key>` is a value the user
        holds, so it comes out of the transcript's own `task_facts` -- the same
        substitution the accumulation line makes
        (`harness.tool_refs.resolve_task_fact_refs`), and the reason it is
        shared rather than re-implemented: a forced transcript's gated write
        names the dictated value exactly the way a gated accumulation step
        does. `$<call_id>.<field>` is a value the world assigned while this
        event ran, so it comes out of the results collected so far.

        A `$task_facts` key the transcript does not carry is an error here
        rather than a literal `"$task_facts.x"` handed to a tool: the ACC
        resolver leaves an unresolved reference in place for its own lints to
        report, and nothing downstream of this point would report it.

        `tool`, when given, is the catalog name of the call being resolved, and
        every resolved argument is then handed over in the JSON type that tool's
        catalog entry declares for it (`_coerced`). Both replays pass it, so the
        offline pin and the live run agree on what the store receives.
        """
        resolved = resolve_task_fact_refs(dict(arguments), self.task_facts)
        unresolved = [
            name
            for name, (producer, _path) in refs_in(resolved).items()
            if producer == TASK_FACTS_PRODUCER
        ]
        if unresolved:
            raise ForcedEventReplayError(
                f"forced-event tool call {call_id!r}: argument(s) {unresolved!r} "
                "name a task_facts key this transcript does not carry: "
                f"task_facts={sorted(self.task_facts)!r}"
            )
        try:
            resolved = resolve_refs(resolved, self.results)
        except ToolRefError as exc:
            raise ForcedEventReplayError(
                f"forced-event tool call {call_id!r}: {exc}"
            ) from exc
        if not tool:
            return resolved
        declared = _catalog_parameter_types().get(
            tool.removeprefix(STATE_TOOL_NAME_PREFIX), {}
        )
        return {
            name: _coerced(value, declared[name]) if name in declared else value
            for name, value in resolved.items()
        }

    def is_held(self, *, call_id: str, name: str, marked_held: bool) -> bool:
        """Whether the gate stops this authored call before it runs.

        The claim is checked, never taken. A hold is only real when this
        session has a gate, the call is on the gate's own tool, and the gate is
        still closed -- exactly the three conditions under which the store's
        lock 1 refuses the same call in an accumulation session. Anything else
        marked `held` is an authoring error, and so is a gated call the writer
        put before the reveal without marking it: live it would reach the world
        and be refused there, which reads as a failed event rather than as the
        hold the session is about.
        """
        closed = not self.gate_is_open()
        held = bool(self.gate) and name == self.gate_tool_name and closed
        if marked_held and not held:
            raise ForcedEventReplayError(
                "forced-event tool call is marked held but nothing holds it: "
                f"call_id={call_id!r} tool={name!r} "
                f"gate_tool={self.gate_tool_name!r} gate_open={not closed}"
            )
        if held and not marked_held:
            raise ForcedEventReplayError(
                "forced-event tool call runs the gate tool before the user has "
                f"dictated its {(self.gate or {}).get('param')!r}: "
                f"call_id={call_id!r}"
            )
        return held

    def observe_message(
        self,
        *,
        source: Any,
        visibility: str,
        reveals: list[str],
        content: str,
    ) -> bool:
        """Take the gate's lock down once the user has dictated the value.

        The accumulation line opens the gate on `beat_enter` of the draft's
        reveal beat (`harness/gate.py:gate_marker_hook`), because there the
        user's words are produced live by the simulator. A forced transcript
        has no beats and its words are frozen, so the reveal is the one
        authored user message the writer bound to the gated fact
        (`MessageItem.reveals`) -- not whichever earlier message happens to
        carry the same terms.

        Three things are checked rather than assumed. The bound message has to
        be one the PA can see: a `visibility: user` message never enters the PA
        session, so a gate opened by it would release the PA's write on
        information the PA was never given. It has to carry the gate's key
        terms by the store's own rule, so both lines agree on what "the user
        said it" means. And it has to come from the user; the schema already
        forbids a `reveals` on any other source, and this is the runtime half
        of the same rule.
        """
        gate = self.gate or {}
        task_fact = str(gate.get("task_fact") or "")
        if not gate or source != "user" or task_fact not in (reveals or []):
            return False
        if visibility not in {"pa", "both"}:
            raise ForcedEventReplayError(
                "the forced-event reveal message must be PA-visible, or the "
                "gate opens on information the PA never received: "
                f"task_fact={task_fact!r} visibility={visibility!r}"
            )
        # Imported here: the store pulls in the whole state server, which the
        # offline generator has no reason to load until it reaches a gate.
        from state_server.store import StateStore

        if not StateStore._carries_key_terms(gate.get("key_terms"), content):
            raise ForcedEventReplayError(
                "forced-event reveal message does not carry the gate's key "
                f"terms: task_fact={task_fact!r} "
                f"key_terms={gate.get('key_terms')!r}"
            )
        self.open_gate()
        return True
