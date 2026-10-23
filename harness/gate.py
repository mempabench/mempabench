"""The session gate: the one write-tool payload the user has to dictate.

A session's gate is a payload parameter of one write tool whose text the user
dictates and only says at the probe beat. Its identity -- which tool, which
parameter, which `task_facts` key, and the key terms the value has to carry --
lives in the persona outline; the text the user actually says lives in the
frozen draft. This module joins the two so the state server can be told, per
step, what it must refuse to accept from the PA.

The probe line (`fin_*` / `pre_*`) has the same gate and a different home for
each half: the identity is a `role: gate` object in the entry's `world` list in
`Authoring/probe/outline/<persona>/outline.yaml`, and the dictated text is in
the frozen probe draft's `task_facts`, exactly where an acc draft keeps it.

It also owns the other marker the harness writes into the state root:
`session_closing`, which shuts a compressed session's final PA turn to every
write tool (`session_closing_hook`).
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from state_server.store import GATE_OPEN_MARKER, SESSION_CLOSING_MARKER

_ACC_SESSION_ID = re.compile(r"^acc_(\d+)$")

_FORCED_EVENT_ID = re.compile(r"^evolv_\d{2}$")

#: A probe session id: `fin_01..28` (2 contexts x 14 attributes) and
#: `pre_01..05` (the five evolving accs), as `scripts/probe_outline_schema.py`
#: mints them. The probe outline keys its entries on this id rather than on an
#: `acc_num`, so it is the identity this module joins the two halves on.
_PROBE_SESSION_ID = re.compile(r"^(?:fin|pre)_\d+$")

#: The acc slots the evolving-event ladder sits on, in ladder order, so
#: `evolv_01..evolv_06` map to 15/30/45/60/75/90. Mirrored from
#: `scripts/session_authoring_schema.py:FORCED_EVENT_ID_BY_ACC` rather than
#: imported (`harness/` does not import from `scripts/`), the same way
#: `COMPRESSED_ATTRIBUTES` below is; `tests/test_harness/test_gate.py` asserts
#: the two are equal, so the table cannot drift out from under this check.
FORCED_EVENT_ID_BY_ACC = {
    acc_num: f"evolv_{position:02d}"
    for position, acc_num in enumerate([15, 30, 45, 60, 75, 90], start=1)
}

#: A catalog tool writes when `allowed_effects` carries one of these -- the rule
#: the draft validator applies too. `verify` is not a write.
_WRITE_EFFECTS = frozenset({"external_write", "internal_write"})

#: The effects of a call that leaves the sandbox -- the act a user releases,
#: as opposed to a write to their own files. `external_action` is here because
#: `_step_effect` admits it on an explicitly-effected step; today's catalog
#: spells every one of them `external_write`.
_EXTERNAL_EFFECTS = frozenset({"external_write", "external_action"})

#: The attributes whose session is authored compressed -- probe beat plus one
#: reaction -- and therefore has no gate to resolve. Mirrored from
#: `scripts/outline_schema.py:COMPRESSED_ATTRIBUTES` rather than imported
#: (`harness/` does not import from `scripts/`), the same way `simulator/schemas.py`
#: carries its own copy; `tests/test_harness/test_gate.py` asserts the two are
#: equal, so the table cannot drift out from under this check.
COMPRESSED_ATTRIBUTES = frozenset(
    {
        "verbosity",
        "tone_formality",
        "emotional_engagement",
        "guidance_level",
        "solution_breadth",
        "uncertainty_expression",
        "task_expansion",
        "capability_boundary",
        "proactive_outreach",
    }
)

#: The two context prefixes a `target_cell` carries, longest first.
_TARGET_CELL_PREFIXES = ("personal_", "work_")


def resolve_session_gate(
    draft_path: Path | str,
    project_root: Path | str,
) -> dict[str, Any] | None:
    """This draft's gate, or None when the session is allowed not to have one.

    A forced `evolv_*` script resolves through the same path as an `acc_*` one:
    the outline now carries an entry for each of the five
    forced accs, two of which declare a `dictated_text` gate, and the artifact
    names that acc itself (`source_acc_num`). Nothing here is FE-specific
    beyond reading the identity off `event` / `event_meta` instead of `meta`.

    A probe (`fin_*` / `pre_*`) resolves through the same path again, reading
    its identity out of the probe outline's entry for its own id instead of the
    persona outline's entry for an acc. Everything downstream -- the marker
    hook, the store's three locks, the simulator's dictation check -- is then
    armed for a probe exactly as it is for an acc, which it was not before:
    `_gate_source` matched only the two acc dialects, so every probe ran with
    `gate=None` and nothing held the PA to eliciting the fact it wrote.

    None is legal in exactly two cases, and both are checked rather than
    assumed: a script that is neither an accumulation session, a ladder event
    nor a probe (a `pretest_*` script, which has no outline entry), and a
    session whose target attribute is compressed -- the probe beat plus one
    reaction, with no task to gate. The compressed table decides it on both
    lines: the probe outline validator gates exactly
    `TAXONOMY - COMPRESSED_ATTRIBUTES`, so a probe carries a gate on exactly
    the attributes an acc session does.

    Everything else raises. `probe.no_gate_reason` is abolished: every
    uncompressed entry carries a dictated_text gate, so an
    uncompressed session with none is a hole, not a gateless session. So are an
    unreadable draft, a missing outline, an outline with no entry for this
    session, and a gated draft whose `task_facts` lacks the key. Each of those
    used to resolve to None, which the state server reads as "this session has
    nothing to enforce" -- a gate that fails open is worse than no gate, because
    the run still looks like it ran gated.
    """
    name = Path(draft_path).name
    draft = _load_yaml(Path(draft_path))
    if not isinstance(draft, dict):
        raise ValueError(f"{name}: not a YAML mapping, so it declares no session")
    source = _gate_source(draft, Path(draft_path))
    if source is None:
        # Neither an accumulation session, a ladder event nor a probe -- a
        # `pretest_*` script. It has no outline entry, so there is nothing to
        # resolve a gate from.
        return None
    meta = source.meta
    persona_id = meta.get("persona_id")
    if not persona_id:
        raise ValueError(f"{name}: a session must name its persona_id")

    outline_path = (
        Path(project_root)
        / ("Authoring/probe/outline" if source.probe_id else "Authoring/outline/drafts")
        / str(persona_id)
        / "outline.yaml"
    )
    if source.probe_id is not None:
        entry = _probe_outline_entry(outline_path, source.probe_id, name)
        ingredient = _probe_gate_ingredient(entry)
    else:
        entry = _outline_entry(outline_path, source.acc_num, name)
        ingredient = _gate_ingredient(entry)
    if ingredient is None:
        _require_gateless_is_legal(meta, name)
        return None

    task_fact = str(ingredient.get("task_fact") or "")
    task_facts = source.task_facts
    if task_fact not in task_facts:
        raise ValueError(
            f"{Path(draft_path).name}: the outline gates on task_fact "
            f"{task_fact!r}, which this draft's task_facts does not carry"
        )
    gate_tool = str(ingredient.get("gate_tool") or "")
    gate_param = str(ingredient.get("gate_param") or "")
    dictated_text = str(task_facts[task_fact])
    if source.probe_id is not None:
        key_terms, gate_cue, gate_slot = _probe_gate_wording(dictated_text)
    else:
        key_terms = _key_terms(ingredient, name)
        gate_cue = _gate_cue(ingredient, name)
        gate_slot = _gate_slot(ingredient)
    return {
        "tool": gate_tool,
        "param": gate_param,
        "task_fact": task_fact,
        "dictated_text": dictated_text,
        "key_terms": key_terms,
        "gate_cue": gate_cue,
        "gate_slot": gate_slot,
        "reveal_beat": _reveal_beat(source.reveal_at, task_fact, Path(draft_path)),
        "pre_reveal_write_tools": _pre_reveal_write_tools(
            draft,
            gate_tool=gate_tool,
            gate_param=gate_param,
            draft_path=Path(draft_path),
            project_root=Path(project_root),
        ),
    }


def resolve_authorised_tool(
    draft_path: Path | str,
    project_root: Path | str,
    *, capability_limit_attempts: list[dict[str, Any]] | None = None,
) -> str | None:
    """The tool this session's user has to release before the PA may run it.

    Read off the plan rather than off a marking: the draft's atomic steps do not
    carry `authorization: required_before_step` -- the authoring schema defaults
    every step to `not_required` and only a gate revealed at a phase after the
    probe is ever required to say otherwise -- so the step that leaves the
    sandbox is what identifies it. That is the same classification the draft
    validator makes, off the catalog's `allowed_effects`.

    Exactly one such tool is expected; repeated calls of it are one act, since
    what the user releases is the tool name and the PA's floor resolves it by
    name. Zero is None; two different tools is a draft to fix rather than a
    choice to make here, so it raises instead of picking. A runtime-bound,
    currently verified no-write capability attempt is excluded; a draft flag
    alone never makes an exception.
    """
    draft = _load_yaml(Path(draft_path))
    if not isinstance(draft, dict):
        raise ValueError(
            f"{Path(draft_path).name}: not a YAML mapping, so it declares no "
            "atomic steps"
        )
    external = _catalog_external_tools(Path(project_root))
    steps = (draft.get("task_execution") or {}).get("atomic_steps") or []
    acting = [
        step
        for step in steps
        if isinstance(step, dict)
        and step.get("actor") == "tool"
        and step.get("tool") in external
    ]
    from harness.capability_attempt import current, matches
    acting = [step for step in acting if not any(
        step.get("expects_capability_limit") is True
        and attempt.get("step_id") == step.get("step_id")
        and matches(attempt | {"arguments": attempt.get("declared_arguments")},
                    step["tool"], step.get("tool_arguments"))
        and matches(attempt, step["tool"], attempt.get("arguments"))
        and current(attempt)
        for attempt in capability_limit_attempts or []
    )]
    tools = {str(step["tool"]) for step in acting}
    if len(tools) > 1:
        named = ", ".join(
            f"{step.get('step_id')} ({step.get('tool')})" for step in acting
        )
        raise ValueError(
            f"{Path(draft_path).name}: {len(tools)} different tools act outside "
            f"the sandbox ({named}); a session releases one tool, so this draft "
            "has to say which"
        )
    return next(iter(tools)) if tools else None


@dataclass(frozen=True)
class _GateSource:
    """One artifact's gate-bearing fields, in the accumulation dialect.

    The forced line keeps the same fields under different keys -- identity in
    `event` (transcript) or `event_meta` (draft) rather than `meta`, and the
    session body nested under `simulation` on a draft -- so this is the one
    place the dialects meet. Everything downstream reads accumulation names,
    which is the rule for this alignment: replicate the ACC mechanism,
    nothing FE-specific.

    Only the fields whose home differs between the dialects are here. The plan
    stays where `_pre_reveal_write_tools` reads it, off the raw draft, because
    the draft validator calls that function directly.

    A probe answers `probe_id` and no `acc_num`: its outline entry is keyed on
    the probe id, and it sits on no acc slot. Exactly one of the two is set,
    which is what `resolve_session_gate` branches the outline lookup on.
    """

    meta: dict[str, Any]
    acc_num: int | None
    task_facts: dict[str, Any]
    reveal_at: dict[str, Any]
    probe_id: str | None = None


def _gate_source(draft: dict[str, Any], path: Path) -> _GateSource | None:
    """Read one artifact's gate-bearing fields, whichever line wrote it.

    None for a script that is neither an accumulation session, a ladder event
    nor a probe -- a `pretest_*` script, say. Those carry no outline entry, so
    there is nothing to resolve a gate from; that was the only true half of the
    early return this replaces.
    """
    meta = draft.get("meta")
    session_id = str((meta or {}).get("session_id") or "")
    acc_match = _ACC_SESSION_ID.match(session_id)
    if isinstance(meta, dict) and acc_match:
        return _GateSource(
            meta=meta,
            acc_num=int(acc_match.group(1)),
            **_body_fields(draft, draft),
        )
    if isinstance(meta, dict) and _PROBE_SESSION_ID.match(session_id):
        # A probe draft, in the accumulation dialect but for its own outline:
        # `task_facts` and `reveal_at` sit at the top level exactly as an acc
        # draft's do, so only the outline it is looked up in differs.
        return _GateSource(
            meta=meta,
            acc_num=None,
            probe_id=session_id,
            **_body_fields(draft, draft),
        )
    if isinstance(meta, dict) and _FORCED_EVENT_ID.match(session_id):
        # A forced artifact written in the accumulation dialect. It names no
        # acc of its own, so the ladder is what says which one it sits on.
        return _GateSource(
            meta=meta,
            acc_num=_acc_num_for_event(session_id, path),
            **_body_fields(draft, draft),
        )
    event = draft.get("event") or draft.get("event_meta")
    if isinstance(event, dict) and _FORCED_EVENT_ID.match(
        str(event.get("event_id") or "")
    ):
        event_id = str(event["event_id"])
        acc_num = _acc_num_for_event(event_id, path)
        declared = event.get("source_acc_num")
        if declared is not None and int(declared) != acc_num:
            raise ValueError(
                f"{path.name}: {event_id} sits on acc {acc_num} in the "
                f"ladder, but source_acc_num says {int(declared)}"
            )
        body = draft.get("simulation")
        return _GateSource(
            meta=event,
            acc_num=acc_num,
            **_body_fields(body if isinstance(body, dict) else draft, draft),
        )
    return None


def _acc_num_for_event(event_id: str, path: Path) -> int:
    """Which acc slot a ladder event sits on."""
    for acc_num, ladder_id in FORCED_EVENT_ID_BY_ACC.items():
        if ladder_id == event_id:
            return acc_num
    raise ValueError(
        f"{path.name}: {event_id} is not a slot of the evolving-event ladder"
    )


def _body_fields(body: dict[str, Any], draft: dict[str, Any]) -> dict[str, Any]:
    """`task_facts` and `reveal_at`, from the body or the top level.

    A forced draft nests its session body under `simulation`; a forced
    transcript and an accumulation draft carry these at the top level.
    """
    return {
        "task_facts": dict(body.get("task_facts") or draft.get("task_facts") or {}),
        "reveal_at": dict(body.get("reveal_at") or draft.get("reveal_at") or {}),
    }


def _key_terms(ingredient: dict[str, Any], name: str) -> list[str]:
    """The information the gated value has to carry.

    The gate is keyed on information, not on words: the
    outline authors these off the dictated sentence and the store passes a
    payload that carries every one of them. An entry that names a gate but no
    terms is a gate the store cannot key on, so it fails closed here rather than
    reaching a live round with nothing to check.
    """
    terms = ingredient.get("gate_key_terms")
    if not isinstance(terms, list) or not terms:
        raise ValueError(
            f"{name}: the outline's gate ingredient carries no gate_key_terms, "
            "so there is no information for the store to key the gate on"
        )
    return [str(term) for term in terms]


def _gate_cue(ingredient: dict[str, Any], name: str) -> str:
    """The phrase that makes asking for the gated information natural.

    The simulator needs it because the actor is the one who has to say it: a cue
    the style retry rewrites away leaves the PA with no reason to ask, which is
    what `gate_cue_absent_before_reveal` refuses at authoring time.

    Fails closed like `_key_terms`, and for the same reason. `gate_kind` is a
    `Literal["dictated_text"]` and the outline schema requires a cue on every
    one of them (`IngredientObject._dictated_text_names_its_cue`), so an
    ingredient reaching here without one is a shape the authoring side already
    forbids. Resolving it to None would arm the actor with nothing while the run
    still looked like it had a motivated gate -- a silent fallback -- so it is
    an error here instead.
    """
    cue = ingredient.get("gate_cue")
    if not isinstance(cue, str) or not cue.strip():
        raise ValueError(
            f"{name}: the outline's gate ingredient carries no gate_cue, so "
            "nothing in the task gives the PA a reason to ask for the gated "
            "information"
        )
    return cue.strip()


def _gate_slot(ingredient: dict[str, Any]) -> str:
    """The slot the dictated value fills, in the author's own words.

    `withheld_fact_before_ask` has to name what the PA asked for, and until this
    field it INFERRED that from the dictated sentence: regress `acc_003` authored
    a note and the judge read it as a reason, so a PA that asked for the right
    thing was scored as if it had not. The slot is a fact about the entry, not a
    reading of the text, so the author declares it.

    Unlike `_key_terms` and `_gate_cue` this does not fail closed. Those two arm
    the lock and the actor; a missing one makes a run LOOK gated while nothing
    holds. The slot is a label carried alongside the gate -- absent, the judge is
    back to the inference it makes today, which is worse but not a false gate.
    Presence is enforced where it can be fixed, at authoring time.
    """
    slot = ingredient.get("gate_slot")
    return slot.strip() if isinstance(slot, str) else ""


def _reveal_beat(
    reveal_at: dict[str, Any], task_fact: str, draft_path: Path
) -> str:
    """The beat at which the user says the gated words.

    Until the session reaches it the store refuses the gate tool outright, so a
    draft that does not say when the reveal happens cannot be run gated: it is a
    broken freeze, the same as one whose `task_facts` lacks the gated key.
    """
    beat = reveal_at.get(task_fact)
    if not beat:
        raise ValueError(
            f"{draft_path.name}: the gate reveals task_fact {task_fact!r}, "
            "which this draft's reveal_at does not place on a beat"
        )
    return str(beat)


def _pre_reveal_write_tools(
    draft: dict[str, Any],
    *,
    gate_tool: str,
    gate_param: str,
    draft_path: Path,
    project_root: Path,
) -> list[str]:
    """The write tools this session is allowed to run before the reveal.

    Everything else is refused while the gate is closed, which is what stops a
    sibling write tool reaching the task's outcome without ever touching the
    gate. "Before" is the gated step's transitive `depends_on` closure, not the
    order the steps happen to be listed in: acc_039 lists its
    `calendar_event_create` hold above the gated `appointment_create`, but both
    hang off the same user choice at the reveal beat, so the hold is not
    something the task needs beforehand -- and reading it as one would hand back
    exactly the bypass this closes.

    An authored forced transcript has no plan to walk: its `pa_tool_call` items
    ARE the plan, and every one the author put before the gated call is a write
    this event runs before the reveal. Same question, read off the artifact the
    forced line actually freezes.

    Takes the raw draft rather than a `_GateSource` because the draft
    validator calls it with `session.model_dump()` to check the frozen draft
    against what the harness would hand the store.
    """
    steps = [
        step
        for step in ((draft.get("task_execution") or {}).get("atomic_steps") or [])
        if isinstance(step, dict)
    ]
    if not steps:
        return _pre_reveal_write_tools_from_transcript(
            draft,
            gate_tool=gate_tool,
            draft_path=draft_path,
            project_root=project_root,
        )
    gated = next(
        (
            step
            for step in steps
            if step.get("tool") == gate_tool
            and gate_param in (step.get("tool_arguments") or {})
        ),
        None,
    )
    if gated is None:
        raise ValueError(
            f"{draft_path.name}: the outline gates {gate_param!r} on "
            f"{gate_tool!r}, which no atomic step calls with that argument"
        )
    depends_on = {
        str(step.get("step_id")): [str(dep) for dep in step.get("depends_on") or []]
        for step in steps
    }
    ancestors: set[str] = set()
    pending = list(depends_on.get(str(gated.get("step_id")), []))
    while pending:
        current = pending.pop()
        if current in ancestors:
            continue
        ancestors.add(current)
        pending.extend(depends_on.get(current, []))
    write_tools = _catalog_write_tools(project_root)
    return sorted(
        {
            str(step.get("tool"))
            for step in steps
            if str(step.get("step_id")) in ancestors
            and str(step.get("tool")) in write_tools
        }
    )


#: How a catalog tool is named in an authored forced transcript: the PA calls
#: it through the state MCP server, which prefixes it. Same rule as
#: `harness/transcript.py:STATE_TOOL_NAME_PREFIX`.
_STATE_TOOL_NAME_PREFIX = "mcp_state_"


def _pre_reveal_write_tools_from_transcript(
    draft: dict[str, Any],
    *,
    gate_tool: str,
    draft_path: Path,
    project_root: Path,
) -> list[str]:
    """The write tools an authored forced transcript calls before its gate."""
    names = [
        str(item.get("name") or "").removeprefix(_STATE_TOOL_NAME_PREFIX)
        for exchange in draft.get("exchanges") or []
        if isinstance(exchange, dict)
        for item in exchange.get("items") or []
        if isinstance(item, dict) and item.get("type") == "pa_tool_call"
    ]
    if gate_tool not in names:
        raise ValueError(
            f"{draft_path.name}: the outline gates on {gate_tool!r}, which no "
            "authored tool call names"
        )
    write_tools = _catalog_write_tools(project_root)
    return sorted(
        {
            name
            for name in names[: names.index(gate_tool)]
            if name in write_tools and name != gate_tool
        }
    )


def _catalog_write_tools(project_root: Path) -> set[str]:
    """Every write tool named by `data/state_tools/catalog.yaml`."""
    return _catalog_tools_with(project_root, _WRITE_EFFECTS)


def _catalog_external_tools(project_root: Path) -> set[str]:
    """Every catalog tool whose call leaves the sandbox."""
    return _catalog_tools_with(project_root, _EXTERNAL_EFFECTS)


def _catalog_tools_with(project_root: Path, effects: frozenset[str]) -> set[str]:
    catalog = _load_yaml(project_root / "data/state_tools/catalog.yaml")
    tools = (catalog or {}).get("tools") or {}
    return {
        name
        for name, spec in tools.items()
        if isinstance(spec, dict)
        and effects & set(spec.get("allowed_effects") or ())
    }


def gate_marker_hook(
    step_runtime: Any,
) -> Callable[[dict[str, Any]], None] | None:
    """Open this step's gate when the session enters its reveal beat.

    The simulator emits `beat_enter` into the harness's transcript writer; this
    is the hook that writer calls, and it is the whole of the first lock. Before
    the marker exists the store refuses the gate tool however good the PA's
    wording is, so a value the user leaked early buys the PA nothing. Returns
    None -- no hook at all -- for every step without a gate.
    """
    gate = getattr(step_runtime, "gate", None) or {}
    reveal_beat = gate.get("reveal_beat")
    if not reveal_beat:
        return None
    marker = Path(step_runtime.state_root) / GATE_OPEN_MARKER

    def open_on_reveal(event: dict[str, Any]) -> None:
        if event.get("kind") == "beat_enter" and event.get("beat") == reveal_beat:
            marker.touch()

    return open_on_reveal


def resolve_compressed_reaction_index(
    draft_path: Path | str,
) -> int | None:
    """Which entered beat is this compressed session's reaction beat, or None.

    Counted the way the simulator loop counts it (`beats_entered ==
    compressed_reaction_index`), because that is the number the store's
    session-closing marker has to agree with. An untriggered compressed session
    runs the probe beat and then the reaction, so the reaction is the second
    beat entered. A triggered one has no probe beat -- the probe was the PA's
    own outreach decision, and `_beats_after_outreach` drops whichever opening
    the decision did not take -- so the opening that plays IS the reaction, and
    it is the first beat entered.

    None for every session that is not compressed, which is every session that
    has a gate: a forced `evolv_*` script, and an `acc_*` session whose target
    attribute is not in `COMPRESSED_ATTRIBUTES`. Those run to their own last
    beat and keep writing to the end.
    """
    draft = _load_yaml(Path(draft_path))
    if not isinstance(draft, dict):
        return None
    meta = draft.get("meta") or {}
    if _ACC_SESSION_ID.match(str(meta.get("session_id") or "")) is None:
        return None
    if _target_attribute(meta, Path(draft_path).name) not in COMPRESSED_ATTRIBUTES:
        return None
    return 1 if draft.get("proactive_trigger") else 2


def session_closing_hook(
    step_runtime: Any,
) -> Callable[[dict[str, Any]], None] | None:
    """Close this compressed session to writes once its last user turn lands.

    The same marker mechanism as `gate_marker_hook`, on the other end of the
    session: the simulator emits `simulator_turn` after the actor has produced
    the reaction beat's message and BEFORE the PA is called with it, so touching
    the marker there leaves it down for exactly the PA turn that answers it --
    the session's last. Returns None -- no hook at
    all -- for every step that is not a compressed session.

    `beat_enter` is the count, and it arrives ahead of the same turn's
    `simulator_turn`: the loop emits it after the actor's first output for the
    beat, along with `goal_already_met`. A collapsed beat records it and no
    turn, which is exactly how the loop counts too. A triggered session's
    outreach turn emits `pa_turn` and no `beat_enter`, so its played opening is
    still the first beat entered.
    """
    reaction_index = getattr(step_runtime, "compressed_reaction_index", None)
    if not reaction_index:
        return None
    marker = Path(step_runtime.state_root) / SESSION_CLOSING_MARKER
    beats_entered = 0

    def close_on_reaction_turn(event: dict[str, Any]) -> None:
        nonlocal beats_entered
        kind = event.get("kind")
        if kind == "beat_enter":
            beats_entered += 1
        elif kind == "simulator_turn" and beats_entered == reaction_index:
            marker.touch()

    return close_on_reaction_turn


def gate_cue_carry_hook(
    script: Any,
) -> Callable[[dict[str, Any]], None] | None:
    """Carry the cue's obligation forward when the simulator skips its beat.

    `scripts/run_orchestrator._arm_dictation_check` pins the cue to one beat and
    the simulator owes it on that beat's opening turn. A beat that runs out of
    turns makes the simulator jump ahead over everything between it and the next
    preference beat (`jump_to_preference_beat_after_turn_limit`), and live
    `acc_003` lost the cue exactly there: `b1_signal_discrepancy` spent
    its three turns, `b2_clarify_context` -- the cue beat -- was skipped, and the
    PA reached the reveal never having been told there was a note to write, so it
    asked for the year line instead.

    The cue is said once or not at all, which is why the jump already refuses to
    skip a `reveal_at` beat. That carve-out lives inside the jump; from here the
    obligation moves instead. Each `beat_skipped` landing on the cue beat
    re-points `gate_cue_beat` at the next beat, and because a jump emits its
    skips in order -- all of them before the beat it landed on runs -- the
    re-point walks the whole skipped run and comes to rest on that landing beat.
    Returns None -- no hook -- for a script that owes its cue on no beat.
    """
    if not getattr(script, "gate_cue_beat", None):
        return None
    beat_ids = [beat.beat for beat in script.beats]

    def carry_on_skip(event: dict[str, Any]) -> None:
        if event.get("kind") != "beat_skipped":
            return
        skipped = event.get("beat")
        if skipped != script.gate_cue_beat:
            return
        # `gate_cue_beat` is validated to be a beat of this script, so the
        # equality above is what makes the lookup below safe.
        after = beat_ids[beat_ids.index(skipped) + 1 :]
        if after:
            script.gate_cue_beat = after[0]

    return carry_on_skip


def _load_yaml(path: Path) -> Any:
    """The file's YAML. A missing or unreadable file is an error, not a None."""
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _outline_entry(outline_path: Path, acc_num: int, name: str) -> dict[str, Any]:
    """The outline entry for this session.

    A persona whose outline is not on disk used to read as "gateless"; it now
    reads as what it is -- the file that says whether this session has a gate
    is missing, so nothing can be said about it.
    """
    if not outline_path.is_file():
        raise FileNotFoundError(
            f"{name}: no outline at {outline_path}, so this session's gate "
            "cannot be resolved"
        )
    outline = _load_yaml(outline_path)
    if not isinstance(outline, dict):
        raise ValueError(f"{name}: {outline_path} is not a YAML mapping")
    for entry in outline.get("sessions") or []:
        if not isinstance(entry, dict):
            continue
        identity = entry.get("identity") or {}
        if identity.get("acc_num") == acc_num:
            return entry
    raise ValueError(
        f"{name}: {outline_path} has no entry with identity.acc_num {acc_num}"
    )


def _gate_ingredient(entry: dict[str, Any]) -> dict[str, Any] | None:
    probe = entry.get("probe") or {}
    for ingredient in probe.get("ingredient_objects") or []:
        if isinstance(ingredient, dict) and ingredient.get("role") == "gate":
            return ingredient
    return None


def _probe_outline_entry(
    outline_path: Path, probe_id: str, name: str
) -> dict[str, Any]:
    """The probe outline's entry for this probe, keyed on its `id`.

    The same three refusals `_outline_entry` makes, on the probe outline's own
    shape (a top-level `probes` list): a missing file, a file that is not a
    mapping, and an entry this probe has none of all read as "nothing can be
    said about this probe's gate" rather than as "no gate".
    """
    if not outline_path.is_file():
        raise FileNotFoundError(
            f"{name}: no probe outline at {outline_path}, so this probe's gate "
            "cannot be resolved"
        )
    outline = _load_yaml(outline_path)
    if not isinstance(outline, dict):
        raise ValueError(f"{name}: {outline_path} is not a YAML mapping")
    for entry in outline.get("probes") or []:
        if isinstance(entry, dict) and entry.get("id") == probe_id:
            return entry
    raise ValueError(f"{name}: {outline_path} has no probe entry with id {probe_id!r}")


def _probe_gate_ingredient(entry: dict[str, Any]) -> dict[str, Any] | None:
    """The entry's `role: gate` world object, or None when it declares none.

    A probe outline spells its ingredients `world` and an acc outline spells
    them `probe.ingredient_objects`; the object itself carries the same
    `gate_tool` / `gate_param` / `gate_kind` / `task_fact` keys
    (`scripts/probe_outline_schema.py:ProbeWorldObject`).
    """
    for ingredient in entry.get("world") or []:
        if isinstance(ingredient, dict) and ingredient.get("role") == "gate":
            return ingredient
    return None


def _probe_gate_wording(dictated_text: str) -> tuple[list[str], str, str]:
    """A probe gate's key terms, cue and slot, off the sentence it gates on.

    The probe outline carries none of the three: `ProbeWorldObject` is the acc
    line's ingredient "minus the acc-only `ref`, `reveal_at` and the key-term /
    cue / slot gate keys", so there is nothing authored to read them from and
    the frozen draft's `task_facts` value is all this side holds.

    * key_terms: the dictated sentence, whole, as one term. The rule is
      that a key term is a word the user would really say and never a token
      invented for the lock, so the only terms that can be named here without
      inventing an author's judgement are the user's own words -- all of them,
      in the order said. That is stricter than a hand-picked set, never looser:
      a payload carrying the sentence carries every term anyone could pick out
      of it. `state_server/store.py:GATE_MIN_KEY_TERMS`, the floor of two for
      hand-picked terms, admits this one term because it is the whole dictated
      text, which the payload then has to carry verbatim.

    * gate_cue: the sentence again. The cue exists to give the PA a reason to
      ask for a fact it will only hear later, and a probe has no later: the one
      active beat reveals the fact in the turn that asks (`ask.say` quotes it,
      and `reveal_at` puts it on that beat). So there is no separate phrase to
      name, and what the field still buys is real -- the actor owes the
      sentence on that beat, and the style retry's "keep these sentences word
      for word" list is built from it. A blank is not an option:
      `simulator/schemas.py:_validate_gate_key_terms` arms the four gate fields
      together or not at all.

    * gate_slot: empty, as for an acc entry whose author left `gate_slot` out.
      It is a label the judge is told rather than a lock, and `_gate_slot` does
      not fail closed on it either.
    """
    return [dictated_text], dictated_text, ""


def _require_gateless_is_legal(meta: dict[str, Any], name: str) -> None:
    """Raise unless this session's attribute is compressed.

    A compressed session is the probe beat plus one reaction, so it has no task
    to gate. That is now the only way to be legitimately gateless:
    `probe.no_gate_reason` is abolished and every
    uncompressed entry carries a dictated_text gate, so an uncompressed session
    with no gate ingredient is a hole that would run ungated while the finished
    run still looked gated.

    The same table rules the probe line: the probe outline validator gates
    exactly `TAXONOMY - COMPRESSED_ATTRIBUTES`, and its A4 row wants exactly one
    `role: gate` object on each of those attributes and none on the rest, so a
    probe is legitimately gateless under exactly this condition too.
    """
    if _target_attribute(meta, name) in COMPRESSED_ATTRIBUTES:
        return
    raise ValueError(
        f"{name}: an uncompressed session declares no gate ingredient, so "
        "nothing holds the PA back"
    )


def _target_attribute(meta: dict[str, Any], name: str) -> str:
    """The attribute half of the draft's `target_cell` (`work_verbosity` etc.).

    A probe draft spells the same field `cell` (`meta.cell` on a
    `probe_session` artifact), which is why both names are read here.
    """
    target_cell = str(meta.get("target_cell") or meta.get("cell") or "")
    for prefix in _TARGET_CELL_PREFIXES:
        if target_cell.startswith(prefix):
            return target_cell[len(prefix) :]
    raise ValueError(
        f"{name}: meta.target_cell {target_cell!r} does not start with a "
        "context prefix, so the target attribute cannot be read"
    )
