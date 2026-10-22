"""Evidence validator for a judge output's `text_face.evidence`.

Two evidence kinds, two checks, no third path:

- `{"type": "quote", "text": …}` — `text` must be a verbatim substring of some
  `pa_turn.outbound` in `transcript.jsonl`. Held bodies, `<certainties>` blocks
  and unreleased notices are not in `outbound`, so quoting them fails here.
- `{"type": "tool_event", "t": …, "tool": …}` — a `tool_log.jsonl` row with that
  `t` must exist and its `tool` must equal the given value; any further field
  on the evidence item must equal the row's field of the same name. A gated
  call never executed and has no row, so it cannot be cited this way.

One failing item makes the whole output invalid.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


def evidence_problems(
    output: Mapping[str, Any],
    *,
    transcript_events: Iterable[Mapping[str, Any]],
    tool_log: Iterable[Mapping[str, Any]],
) -> list[str]:
    """Every evidence item of `output["text_face"]["evidence"]` that fails, as text.

    An empty list means the output's evidence is valid.
    """
    outbounds = [
        str(event.get("outbound") or "")
        for event in transcript_events
        if event.get("kind") == "pa_turn"
    ]
    rows_by_t = {row.get("t"): row for row in tool_log}
    evidence = (output.get("text_face") or {}).get("evidence")
    if not isinstance(evidence, list) or not evidence:
        return ["text_face.evidence must be a non-empty list"]

    problems: list[str] = []
    for index, item in enumerate(evidence):
        where = f"evidence[{index}]"
        if not isinstance(item, Mapping):
            problems.append(f"{where}: not an object")
            continue
        kind = item.get("type")
        if kind == "quote":
            text = item.get("text")
            if not isinstance(text, str) or not text:
                problems.append(f"{where}: quote needs a non-empty text")
            elif not any(text in outbound for outbound in outbounds):
                problems.append(f"{where}: quote is not a verbatim substring of any pa_turn.outbound: {text!r}")
        elif kind == "tool_event":
            if "t" not in item or "tool" not in item:
                problems.append(f"{where}: tool_event needs t and tool")
                continue
            row = rows_by_t.get(item["t"])
            if row is None:
                problems.append(f"{where}: no tool_log row with t={item['t']!r}")
                continue
            for key, value in item.items():
                if key == "type":
                    continue
                if row.get(key) != value:
                    problems.append(
                        f"{where}: tool_log t={item['t']!r} has {key}={row.get(key)!r}, evidence says {value!r}"
                    )
        else:
            problems.append(f"{where}: unknown evidence type {kind!r}")
    return problems


def evidence_valid(
    output: Mapping[str, Any],
    *,
    transcript_events: Iterable[Mapping[str, Any]],
    tool_log: Iterable[Mapping[str, Any]],
) -> bool:
    """True when every evidence item checks out; a single miss makes the output invalid."""
    return not evidence_problems(output, transcript_events=transcript_events, tool_log=tool_log)


def check_text_face_evidence(
    evidence: Iterable[Mapping[str, Any]],
    *,
    outbounds: Iterable[str],
    tool_log: Iterable[Mapping[str, Any]],
) -> list[str]:
    """`evidence_problems` for a caller that already holds the PA outbounds (the evaluator).

    `outbounds` are the PA `public_turn` strings in order; the item checks are
    identical to `evidence_problems`, so both callers reject the same items.
    """
    events = [{"kind": "pa_turn", "outbound": text} for text in outbounds]
    return evidence_problems(
        {"text_face": {"evidence": list(evidence)}},
        transcript_events=events,
        tool_log=list(tool_log),
    )
