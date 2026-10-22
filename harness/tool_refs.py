"""Explicit references from one tool call's arguments to an earlier result.

Drafts and transcripts cannot know the ids the world assigns while a session
runs (``draft_0007``, ``res_0003`` ...). Instead of guessing, an argument
names the producer and the field it wants::

    booking_id: $pa_book_room.booking_id
    email_id:   $search_inbox.matches.0.email_id
    email_id:   $int_reyes_reply.id          # a delivered environment message

The producer is a draft ``step_id`` (in session/event drafts) or a transcript
``call_id`` / ``action_id`` (in forced transcripts). The path is dotted; a
numeric segment indexes a list.

The path may END in a selector instead of an index, ``[<state-relative path>]``::

    path: $list_bistro_folder.files[documents/bistro/tipout_and_paystubs.md]

which picks the one element of a returned string list equal to that text
(owner, 2026-09-20). A folder listing is where a fixed index is a structural
error rather than an off-by-one: an earlier session adds a file to the folder,
and the PA names its own writes at run time, so the position of the file the
draft means moves between the L0 world and the run. At most one selector, only
at the end, and it is a single path segment -- it is NOT split on ``.``, or a
``.md`` suffix would become a segment of its own. The text inside the brackets
is a plain state-relative path: no space, no wildcard, no ``..``. Nothing is
composed, looked for on disk, or fallen back to: no match and more than one
match are both errors.
"""

from __future__ import annotations

import re
from typing import Any

#: What may sit inside a selector's brackets: a state-relative path. Written as
#: dot-free runs joined by single dots, so `a.md` is in and `../a.md` is out --
#: two dots can never meet, and a traversal cannot be written at all.
_SELECTOR_BODY = r"[A-Za-z0-9_\-/]+(?:\.[A-Za-z0-9_\-/]+)*"

REF_RE = re.compile(
    r"^\$([A-Za-z][A-Za-z0-9_\-]*)"
    r"\.([A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)*)"
    rf"(\[{_SELECTOR_BODY}\])?$"
)

#: A path segment that SELECTS by value rather than by position:
#: ``[documents/bistro/tipout_and_paystubs.md]``. The brackets are kept on the
#: segment `parse_ref` hands back, which is what tells the walk (and the
#: catalog-stamp writer) a selector apart from an ordinary key.
SELECTOR_RE = re.compile(rf"^\[({_SELECTOR_BODY})\]$")

#: A value that is *trying* to be a reference: the sigil, a producer name and a
#: dotted path. Anything short of this is ordinary text and is left alone -- the
#: drafts are full of prices (``$45.00``, ``$$$``) that are not failed
#: references. A value that matches this but not `REF_RE` is a broken
#: reference, and broken is an error rather than a silent literal.
REF_ATTEMPT_RE = re.compile(r"^\$[A-Za-z][A-Za-z0-9_\-]*\..")

#: A `$task_facts.<path>` reference sitting INSIDE a longer string: the user's
#: dictated sentence is written into a body the draft composes around it -- the
#: anchored line a tail-append follows, the elicited fields, the rest of a note.
#: Only `task_facts` embeds. A step result (`$step.field`) is an id the world
#: mints and has no larger string to sit in, and admitting the general shape
#: here would turn every draft's `$45.00` into a reference attempt again.
TASK_FACT_EMBED_RE = re.compile(
    r"\$task_facts\.([A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)*)"
)


class ToolRefError(ValueError):
    """A ``$producer.path`` reference cannot be resolved."""


def parse_ref(value: Any) -> tuple[str, list[str]] | None:
    if not isinstance(value, str):
        return None
    match = REF_RE.match(value.strip())
    if match is None:
        return None
    path = match.group(2).split(".")
    if match.group(3):
        # One more segment, brackets included, NOT split on `.`: the whole
        # bracketed text is the value to select by.
        path.append(match.group(3))
    return match.group(1), path


def is_path_selector(segment: str) -> bool:
    """`segment` selects a list element by value: ``[<state-relative path>]``."""
    return SELECTOR_RE.fullmatch(segment) is not None


def _path_text(path: list[str]) -> str:
    """The path written the way the author wrote it.

    Dotted, except that a selector segment is glued on without a dot --
    ``files[documents/bistro/tipout_and_paystubs.md]``, not ``files.[...]``,
    which would not parse again.
    """
    written = "".join(
        segment if is_path_selector(segment) else f".{segment}" for segment in path
    )
    return written[1:] if written.startswith(".") else written


def embedded_task_fact_paths(value: Any) -> list[list[str]]:
    """The `task_facts` paths a longer string embeds, in the order they appear.

    Empty for a value that is nothing but a reference: that one is a whole-value
    reference, which `parse_ref` already reads and which resolves to the fact's
    own value rather than to its string form.
    """
    if not isinstance(value, str) or parse_ref(value) is not None:
        return []
    return [match.group(1).split(".") for match in TASK_FACT_EMBED_RE.finditer(value)]


def _ref_sites(label: str, value: Any) -> dict[str, tuple[str, list[str]]]:
    """Every reference inside one argument value, by the label that names it.

    The label is the argument name for a plain value and ``name.<index>`` for a
    list element, so the two reference sites an argument may hold stay apart in
    the mapping and each can be written back where it was found.

    Lists are walked because an argument may BE a list of references:
    ``email_message_send.attachments`` is an array of state-relative paths, and
    the path an earlier read step returned is the only way to name the file
    without writing the path out (owner, 2026-09-17). Dicts are not walked --
    no catalog parameter takes one -- so a mapping-valued argument stays a
    literal exactly as it was.
    """
    if isinstance(value, list):
        found: dict[str, tuple[str, list[str]]] = {}
        for index, item in enumerate(value):
            found.update(_ref_sites(f"{label}.{index}", item))
        return found
    parsed = parse_ref(value)
    if parsed is not None:
        return {label: parsed}
    if embedded_task_fact_paths(value):
        # A template string: `$task_facts.<key>` plus the body written around
        # it. It is not a reference-valued argument -- the whole value is not
        # the reference -- so it is not returned here; `task_fact_refs` reads
        # it, and `resolve_task_fact_refs` substitutes it in place. Skipped
        # before the attempt check below, which a template starting with the
        # reference would otherwise reject as a broken one.
        return {}
    if isinstance(value, str) and REF_ATTEMPT_RE.match(value.strip()):
        raise ToolRefError(
            f"argument {label!r}: {value!r} reads as a reference but does not "
            "parse. A path segment may only be a name (letters, digits, "
            "underscore) or a list index; a key with a space or punctuation "
            "in it cannot be referenced -- have the producing step return it "
            "under a referenceable key, or pass the value literally."
        )
    return {}


def _map_ref_values(value: Any, replace: Any) -> Any:
    """Rebuild `value`, applying `replace` to every site `_ref_sites` reads."""
    if isinstance(value, list):
        return [_map_ref_values(item, replace) for item in value]
    return replace(value)


def refs_in(arguments: dict[str, Any]) -> dict[str, tuple[str, list[str]]]:
    """Reference site -> (producer, path) for every reference an argument holds.

    Keyed by the argument name, or by ``name.<index>`` for a reference sitting
    in a list element.

    A value that reads as a reference but does not parse raises. It used to be
    skipped, which sent the author's reference to the world as literal text:
    ``$s.matches.0.nightly_rates.ocean view`` -- the user_c hotel fixtures label
    a rate with a space -- arrived as that whole string and matched nothing,
    with nothing said. There is deliberately no quoted-key syntax to reach such
    a key; the error says so instead.
    """
    found: dict[str, tuple[str, list[str]]] = {}
    for name, value in arguments.items():
        found.update(_ref_sites(name, value))
    return found


def _selected_path(current: Any, wanted: str) -> str:
    """The one element of a returned string list that EQUALS `wanted`.

    Whole-string equality over the list the step really handed back. The
    filesystem is never consulted, no path is composed, and neither a miss nor
    a tie falls back to an element: the point of the selector is that the draft
    names the file it means, so anything but exactly one match is an error the
    author has to see. The message prints the list it actually got, because
    "the file is not there" and "the file is there under another name" are the
    two diagnoses and only the list separates them.
    """
    if not isinstance(current, list) or not all(
        isinstance(item, str) for item in current
    ):
        raise ToolRefError(
            f"path selector [{wanted}] needs a list of strings to choose from, "
            f"got {current!r}"
        )
    hits = [item for item in current if item == wanted]
    if len(hits) == 1:
        return hits[0]
    if not hits:
        raise ToolRefError(
            f"path selector [{wanted}] matches none of the {len(current)} paths "
            f"the step returned: {current!r}"
        )
    raise ToolRefError(
        f"path selector [{wanted}] matches {len(hits)} of the paths the step "
        f"returned, so which one is meant is ambiguous: {current!r}"
    )


def _walk(result: Any, path: list[str]) -> Any:
    current = result
    for segment in path:
        selector = SELECTOR_RE.fullmatch(segment)
        if selector is not None:
            current = _selected_path(current, selector.group(1))
        elif isinstance(current, list):
            if not segment.isdigit() or int(segment) >= len(current):
                raise KeyError(segment)
            current = current[int(segment)]
        elif isinstance(current, dict):
            if segment not in current:
                raise KeyError(segment)
            current = current[segment]
        else:
            raise KeyError(segment)
    return current


def resolve_refs(
    arguments: dict[str, Any], results: dict[str, Any]
) -> dict[str, Any]:
    """Replace every ``$producer.path`` reference with the referenced value.

    A reference in a list element resolves in place; the list is rebuilt rather
    than written through, so the caller's own list is left as it was.
    """
    resolved = dict(arguments)
    for name, value in arguments.items():
        if not _ref_sites(name, value):
            # Nothing to resolve here -- and a value that reads as a broken
            # reference has already raised out of `_ref_sites`.
            continue
        resolved[name] = _map_ref_values(
            value, lambda item: _resolved_value(name, item, results)
        )
    return resolved


def _resolved_value(name: str, value: Any, results: dict[str, Any]) -> Any:
    parsed = parse_ref(value)
    if parsed is None:
        return value
    producer, path = parsed
    if producer not in results:
        raise ToolRefError(
            f"argument {name!r} refers to {producer!r}, which has not produced "
            "a result yet"
        )
    try:
        return _walk(results[producer], path)
    except ToolRefError as exc:
        raise ToolRefError(f"argument {name!r}: {producer!r} result: {exc}") from exc
    except KeyError as exc:
        raise ToolRefError(
            f"argument {name!r}: {producer!r} result has no field "
            f"{_path_text(path)!r} (missing {exc.args[0]!r}); "
            f"result={results[producer]!r}"
        ) from exc


#: The producer name a `tool_arguments` value uses to point at a `task_facts`
#: key rather than at an earlier step's result: `$task_facts.<key>`. `REF_RE`
#: above already parses this shape, so the syntax is shared with
#: `$<step_id>.<field>`; only the meaning differs -- the value is one the user
#: holds, not one an earlier call produced. It lives here, beside the syntax it
#: reuses, because three lines read it: the accumulation draft validator, the
#: forced-event executor, and the forced-event offline replay.
TASK_FACTS_PRODUCER = "task_facts"

_MISSING_FACT = object()


def task_fact_refs(arguments: dict[str, Any] | None) -> dict[str, list[str]]:
    """Argument name -> the `task_facts` path a `$task_facts.…` argument names.

    The link the gating lints need: which fact this tool call consumes, stated
    rather than guessed from whether the fact's value happens to be a substring
    of the arguments. The path may descend into a nested fact
    (`$task_facts.file_locations.draft_introduction`); its FIRST segment is the
    `task_facts` key, which is the granularity `reveal_at` works at.

    An argument may also EMBED the reference in a longer string (owner,
    2026-09-11): the draft writes the body around `$task_facts.<key>` -- the
    anchored line, the elicited fields, the rest of a note. That argument
    consumes the fact exactly as a whole-value reference does, so it is
    reported here too, by the first path it embeds.
    """
    found = {
        label: path
        for label, (producer, path) in refs_in(arguments or {}).items()
        if producer == TASK_FACTS_PRODUCER
    }
    for name, value in (arguments or {}).items():
        found.update(_embedded_sites(name, value))
    return found


def _embedded_sites(label: str, value: Any) -> dict[str, list[str]]:
    """Reference site -> the first `task_facts` path a template there embeds."""
    if isinstance(value, list):
        found: dict[str, list[str]] = {}
        for index, item in enumerate(value):
            found.update(_embedded_sites(f"{label}.{index}", item))
        return found
    paths = embedded_task_fact_paths(value)
    return {label: paths[0]} if paths else {}


def _fact_at(task_facts: dict[str, Any], path: list[str]) -> Any:
    """The value a `$task_facts` path names, or `_MISSING_FACT`."""
    current: Any = task_facts
    for segment in path:
        if not isinstance(current, dict) or segment not in current:
            return _MISSING_FACT
        current = current[segment]
    return current


def resolve_task_fact_refs(
    arguments: dict[str, Any], task_facts: dict[str, Any]
) -> dict[str, Any]:
    """Substitute `$task_facts.…` arguments with the fact's literal value.

    Runs before `resolve_refs`, which resolves step references against results
    the dry run collected and would otherwise reject `task_facts` as a producer
    that has not run.

    A whole-value reference resolves to the fact itself, whatever its type. An
    EMBEDDED reference resolves in place, inside the string the draft wrote
    around it, so the fact arrives as text in the body that carries it; a fact
    the mapping does not hold is left standing rather than blanked, exactly as
    a whole-value reference to a missing fact is left alone here.
    """
    resolved = dict(arguments)
    for name, value in arguments.items():
        # A value that reads as a broken reference raises here, as it did when
        # this loop went through `task_fact_refs`.
        _ref_sites(name, value)
        resolved[name] = _map_ref_values(
            value, lambda item: _resolved_fact_value(item, task_facts)
        )
    return resolved


def _resolved_fact_value(value: Any, task_facts: dict[str, Any]) -> Any:
    if embedded_task_fact_paths(value):
        return TASK_FACT_EMBED_RE.sub(
            lambda match: _embedded_substitution(match, task_facts), value
        )
    parsed = parse_ref(value)
    if parsed is None or parsed[0] != TASK_FACTS_PRODUCER:
        return value
    fact = _fact_at(task_facts, parsed[1])
    return value if fact is _MISSING_FACT else fact


def _embedded_substitution(match: re.Match[str], task_facts: dict[str, Any]) -> str:
    fact = _fact_at(task_facts, match.group(1).split("."))
    return match.group(0) if fact is _MISSING_FACT else str(fact)


def rewrite_ref_producers(
    arguments: dict[str, Any], mapping: dict[str, str]
) -> dict[str, Any]:
    """Rename reference producers (draft step ids -> transcript call ids)."""
    rewritten = dict(arguments)
    for name, value in arguments.items():
        if not _ref_sites(name, value):
            continue
        rewritten[name] = _map_ref_values(
            value, lambda item: _rewritten_producer(item, mapping)
        )
    return rewritten


def _rewritten_producer(value: Any, mapping: dict[str, str]) -> Any:
    parsed = parse_ref(value)
    if parsed is None or parsed[0] not in mapping:
        return value
    return f"${mapping[parsed[0]]}.{_path_text(parsed[1])}"
