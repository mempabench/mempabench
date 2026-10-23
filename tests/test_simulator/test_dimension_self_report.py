"""Dimension self-report: prompt assembly, the rebuild of the given blocks from
recorded probe data, the output parser, and the caller. The probe data is
synthetic, over user_b's public drafts and persona files. No network: every
provider here is scripted, and the one real provider class has its client
replaced."""

from __future__ import annotations

import asyncio
import itertools
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

import simulator.dimension_self_report as dsr
from simulator.dimension_self_report import (
    DimensionSelfReporter,
    RebuildError,
    build_prompt,
    character_card,
    load_prompt_files,
    parse_report,
    rebuild_blocks,
)
from simulator.judge.catalogue import OUTREACH_ACT
from simulator.profiles import load_simulator_profile
from simulator.prompting import TRIGGERED_OPENING_NOTE, WITHHELD_FACTS_LINE
from simulator.providers.base import LLMProvider, LLMResponse
from simulator.providers.openai_compat_provider import OpenAICompatProvider
from simulator.schemas import PreferenceMatrix, SessionScript
from simulator.self_report import SELF_REPORT_MAX_TOKENS
from simulator.skills import normalize_setting

ROOT = Path(__file__).resolve().parents[2]
DRAFTS = ROOT / "Authoring/probe/drafts/user_b"
PREFERENCES = ROOT / "data/personas/user_b_preferences.yaml"
SPECS = ROOT / "data/actor_skill_specs"
CLOCK = datetime(2026, 2, 3, 9, 30)
WRAPPER = {"notices_released": 0, "completion_released": False, "confidence_brackets": False}
VALID = (
    "<dimension_self_report>\nscore: 4\n"
    "reason: In turn 1 it said 'Done' and that suited me.\n</dimension_self_report>"
)


# ----------------------------------------------------------------- helpers


def _profile(tmp_path: Path, memory: str | None = None):
    root = tmp_path / "user_b"
    root.mkdir(parents=True)
    shutil.copy(ROOT / "data/personas/user_b_identity.yaml", root / "identity.yaml")
    shutil.copy(PREFERENCES, root / "preferences.yaml")
    shutil.copy(ROOT / "simulator/profiles/user_b/TOOLS.md", root / "TOOLS.md")
    (root / "profile.yaml").write_text(
        "identity: identity.yaml\npreference_matrix: preferences.yaml\nmemory:\n  mode: nanobot\n",
        encoding="utf-8",
    )
    if memory is not None:
        (root / "memory").mkdir()
        (root / "memory" / "MEMORY.md").write_text(memory, encoding="utf-8")
    return load_simulator_profile(root)


def _script(pid: str) -> SessionScript:
    return SessionScript.from_authored_session_yaml(DRAFTS / pid / f"{pid}.yaml")


def _rebuild(tmp_path, pid, attribute, events, derived, overlay=None):
    return rebuild_blocks(
        events,
        script=_script(pid),
        session_clock=CLOCK,
        profile=_profile(tmp_path),
        preferences=PreferenceMatrix.from_yaml(PREFERENCES),
        recorded_overlay=overlay,
        derived=derived,
        attribute=attribute,
        specs_root=SPECS,
    )


def _enter(t, beat):
    return {"kind": "beat_enter", "t": t, "beat": beat}


def _me(t, beat, message):
    return {"kind": "simulator_turn", "t": t, "beat": beat, "message": message}


def _pa(t, beat, outbound, tools=(), **extra):
    return {"kind": "pa_turn", "t": t, "beat": beat, "outbound": outbound, "tool_events": list(tools), **extra}


def _derived(*ts, held=None):
    return {t: {"reply_facts": dict(WRAPPER), "held_items": list(held or [])} for t in ts}


def _autonomy_probe():
    """fin_08: one beat, two exchanges."""
    events = [
        {"kind": "session_start", "t": 0},
        _enter(1, "ask"),
        _me(2, "ask", "Can you sort out the Studio B booking?"),
        _pa(3, "ask", "I can book it. Shall I go ahead?", [{"name": "mcp_state_calendar_event_search"}]),
        _me(4, "ask", "Yes, go ahead."),
        _pa(5, "ask", "Booked.", [{"name": "mcp_state_contact_update"}]),
        {"kind": "session_end", "t": 6},
    ]
    return events, _derived(3, 5)


def _outreach_probe(delivered: bool):
    """fin_09: the PA's own trigger turn, then the opening that matches it."""
    opening = "opening_after_send" if delivered else "opening_after_suppress"
    skipped = "opening_after_suppress" if delivered else "opening_after_send"
    events = [
        {"kind": "environment_trigger", "t": 1, "detail": "Gwen's email arrived"},
        _pa(
            2,
            "proactive_trigger",
            "Gwen emailed about the showcase." if delivered else "",
            [{"name": "IX_proactive_outreach", "ix": True}, {"name": "mcp_state_email_message_search"}],
            proactive_event=True,
            outreach_delivered=delivered,
        ),
        {"kind": "outreach_path", "t": 3, "path": "send" if delivered else "suppress",
         "opening_beat": opening, "skipped_beat": skipped},
        _enter(4, opening),
        _me(5, opening, "I saw it."),
        _pa(6, opening, "Noted."),
    ]
    return events, _derived(2, 6)


def _prompt(rebuilt, attribute):
    return build_prompt(rebuilt.blocks, attribute, rebuilt.setting)


# ------------------------------------------------------- prompt assembly


def _blocks(attribute="verbosity", setting="Terse"):
    return {
        "CHARACTER_CARD": "card",
        "ACTING_SHEET": f"preamble\n\n---\n\n## Skill: {attribute} = {setting}\n\nbody",
        "LIFE_CONTEXT": "life",
        "SESSION_SETUP": "setup",
        "CONVERSATION": "turn 0 — me:\n<<QUESTION>> stays as typed",
        "PRIVATE_NOTES": "notes",
    }


def test_frame_fills_in_order_with_the_settings_own_who_row() -> None:
    _, attributes = load_prompt_files()
    entry = attributes["verbosity"]
    prompt = build_prompt(_blocks(), "verbosity", "Terse")

    assert (
        prompt.index("# THE CONVERSATION")
        < prompt.index("# AN EXAMPLE FROM SOMEONE ELSE")
        < prompt.index(entry["example"])
        < prompt.index("# QUESTION")
        < prompt.index("# WHAT EACH SCORE MEANS")
    )
    assert entry["question"].replace("{who}", entry["who"]["Terse"]) in prompt
    assert entry["who"]["Detailed"] not in prompt.split("# QUESTION")[1]
    anchors = prompt.split("# WHAT EACH SCORE MEANS\n\n")[1].split("\n\nOutput exactly")[0]
    assert anchors.split("\n") == [f"{s}: {entry['anchors'][s]}" for s in (5, 4, 3, 2, 1)]
    # One pass: a block's own text is never read as a placeholder.
    assert "<<QUESTION>> stays as typed" in prompt
    assert prompt.count("<<") == 1


def test_a_sheet_for_another_setting_is_refused() -> None:
    with pytest.raises(RebuildError, match="acting sheet"):
        build_prompt(_blocks(setting="Detailed"), "verbosity", "Terse")


def test_every_actor_skill_spec_has_one_who_row_and_every_attribute_is_complete() -> None:
    _, attributes = load_prompt_files()
    specs = {(path.parent.name, path.stem) for path in SPECS.glob("*/*.md")}
    rows = {(attr, normalize_setting(setting)) for attr, entry in attributes.items() for setting in entry["who"]}
    assert rows == specs
    for entry in attributes.values():
        assert entry["question"].count("{who}") == 1
        assert list(entry["anchors"]) == [5, 4, 3, 2, 1]
        assert entry["example"].count("<dimension_self_report>") == 2


# ---------------------------------------------------------------- rebuild


def test_character_card_is_build_context_without_profile_memory(tmp_path: Path) -> None:
    plain = _profile(tmp_path / "plain")
    with_memory = _profile(tmp_path / "memory", memory="MEMORY-SENTINEL")

    assert "## Profile Memory" in with_memory.build_context()
    assert character_card(with_memory) == plain.build_context()
    assert "MEMORY-SENTINEL" not in character_card(with_memory)


_KEEP = {("pa_turn", "outbound"), ("simulator_turn", "message"), ("simulator_turn", "beat"),
         ("beat_enter", "beat"), ("outreach_path", "skipped_beat")}
#: Numbers and flags the rebuild reads: kept as they are. Every other number
#: and flag is changed. `error` keeps its flag; any text inside it is planted.
_KEEP_VALUES = {("pa_turn", "t"), ("pa_turn", "proactive_event"), ("pa_turn", "outreach_delivered"),
                ("simulator_turn", "t")}
_KEEP_TOOL_VALUES = {"ix", "ix_event", "error", "process"}


def _plant(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every string outside the rendered fields becomes its own sentinel, every
    number outside them `900000 + n`, and every flag outside them flips."""
    counter = itertools.count()

    def fill(value, keep_value=False):
        if isinstance(value, str):
            return f"SENTINEL-{next(counter)}" if value else value
        if isinstance(value, dict):
            return {key: fill(item) for key, item in value.items()}
        if isinstance(value, list):
            return [fill(item) for item in value]
        if keep_value or value is None:
            return value
        if isinstance(value, bool):
            return not value
        return 900000 + next(counter)

    planted = []
    for event in events:
        kind = event["kind"]
        out = {}
        for key, value in event.items():
            if key == "kind" or (kind, key) in _KEEP:
                out[key] = value
            elif key == "tool_events":
                out[key] = [
                    {k: v if k == "name" else fill(v, k in _KEEP_TOOL_VALUES) for k, v in item.items()}
                    for item in value
                ]
            else:
                out[key] = fill(value, (kind, key) in _KEEP_VALUES)
        planted.append(out)
    return planted


def test_only_whitelisted_fields_reach_the_prompt(tmp_path: Path) -> None:
    events, derived = _autonomy_probe()
    events[1].update(
        cued_ix_tools=["IX_autonomy_level"], active_ix_tools=["IX_autonomy_level"], elicitation="x"
    )
    events[2].update(
        factual_check="checked",
        turn_assessment={"next_beat": "stay", "goal_already_met": False},
        consent_decisions=[{"interaction_item_id": "held_1", "decision": "allow"}],
        parse_warnings=["w"],
    )
    events[3].update(
        metadata={"ix_render": {
            "held": {"outreach_message": "held text"},
            "proactive_decision": {"setting": "Low"},
        }},
        ix_required=["IX_autonomy_level"], ix_called=["IX_autonomy_level"], ix_missing=[], chat_id="c",
        benchmark_turn_timestamp="2026-02-03T09:31:00", empty=False,
    )
    events[3]["tool_events"] += [
        {"name": "IX_autonomy_level", "ix": True, "attribute": "autonomy_level", "setting": "Reactive",
         "evidence": "e", "payload_type": "p"},
        {"name": "ix_hold", "ix_event": True, "detail": "held", "tool": "email_send"},
        {"process": True, "visibility": "bookend", "stage": "start", "message": "Starting now"},
        {"name": "mcp_state_contact_read", "status": "capability_limit", "call_index": 4,
         "error": {"ok": False, "error_code": "capability_limit", "detail": "held in the portal"}},
    ]
    events[3]["tool_events"][0].update(args={"query": "Studio B"}, detail="found", status="ok")
    events[5]["tool_events"][0].update(args={"id": "c_1"}, error="")  # an empty error is no error
    events[5:5] = [
        {"kind": "judge_call", "t": 50, "rule": "r",
         "answers": [{"evidence": "e", "reason": "r"}], "verdicts": ["v"]},
        {"kind": "tool_arguments_off_plan", "t": 51, "args": {"to": "a@b.example", "body": "b"}},
        {"kind": "tool_outside_plan", "t": 52, "tool": "email_send"},
        {"kind": "simulator_turn_diagnostics", "t": 53, "diagnostics_detail": "beat instruction quoted"},
        {"kind": "consent_preauthorised", "t": 54, "tool": "t"},
    ]
    events.append({
        "kind": "session_end_reflection", "t": 55, "reflection": "r",
        # The actor's earlier session-end answers: mostly numbers.
        "self_report": {"felt_understood_in_this_context": 5, "felt_effort": 1, "overall_experience": 5,
                        "noticed_contradictions": "n"},
        "evidence": [{"sentence": "s", "turn": 1, "t": 3}],
    })
    next(e for e in events if e["kind"] == "session_end").update(simulator_turns=2, beat_count=1)
    derived[3]["held_items"] = [{"interaction_item_id": "HELD-ID-SENTINEL", "intent": "book Studio B"}]

    clean = _prompt(_rebuild(tmp_path / "clean", "fin_08", "autonomy_level", events, derived), "autonomy_level")
    rebuilt = _rebuild(tmp_path / "planted", "fin_08", "autonomy_level", _plant(events), derived)
    prompt = _prompt(rebuilt, "autonomy_level")

    assert "SENTINEL" not in prompt
    # Numbers and flags outside the read fields were changed too, so a leaked
    # score or flag would make the two prompts differ.
    assert _plant(events)[-1]["self_report"]["overall_experience"] != 5
    assert prompt == clean
    # The rendered fields did get through, so the check is not vacuous.
    for kept in ("Can you sort out the Studio B booking?", "Booked.", "- book Studio B",
                 "contact_update — update one existing contact by identifier"):
        assert kept in prompt
    # Not given: the message template, profile memory, the naturalness guide, the per-turn
    # instructions, the reply-facts frame, the held-items instruction, IX names.
    for literal in ("## Profile Memory", "# GLOBAL RULES", "# USER DIALOGUE NATURALNESS",
                    "# User Dialogue Naturalness", "Write the user's visible message as something",
                    "Produce only the current", "Output exactly three XML-like blocks",
                    "Private record of what the assistant's latest reply",
                    "A change the reply describes but this record does not list",
                    "Then add a <consent_decisions> block", "IX_"):
        assert literal not in prompt, literal


def test_a_tool_error_is_read_as_a_flag_only() -> None:
    read = dsr._read({"kind": "pa_turn", "t": 1, "tool_events": [
        {"name": "a", "error": {"ok": False, "error_code": "capability_limit", "detail": "held in the portal"}},
        {"name": "b", "error": ""},
        {"name": "c"},
    ]})
    assert read["tool_events"] == [{"name": "a", "error": True}, {"name": "b", "error": False}, {"name": "c"}]


def test_process_records_are_not_counted_as_tool_calls(tmp_path: Path) -> None:
    events, derived = _autonomy_probe()
    events[3]["tool_events"] += [{"process": True, "stage": "start"}, {"process": True, "stage": "completion"}]
    line = "- Tool calls it made for the task in that reply (counting a call that came back blocked or failed): {}"

    notes = _rebuild(tmp_path, "fin_08", "autonomy_level", events, derived).blocks["PRIVATE_NOTES"]
    assert line.format(1) in notes.split("\n\n")[0]


def test_notes_follow_every_delivered_reply_numbered_by_turn(tmp_path: Path) -> None:
    events, derived = _autonomy_probe()
    rebuilt = _rebuild(tmp_path, "fin_08", "autonomy_level", events, derived)

    notes = rebuilt.blocks["PRIVATE_NOTES"].split("\n\n")
    assert [n.split("\n")[0] for n in notes] == ["private notes after turn 1:", "private notes after turn 3:"]
    assert all(len(n.split("\n")) == 7 for n in notes)
    assert notes[1].split("\n")[1] == (
        "- Changes it carried out in that reply: "
        "contact_update — update one existing contact by identifier"
    )
    assert notes[1].split("\n")[2] == "- Changes it had already carried out earlier in this conversation: none"


def test_held_items_show_their_description_and_never_the_id(tmp_path: Path) -> None:
    events, derived = _autonomy_probe()
    derived[3]["held_items"] = [
        {"interaction_item_id": "ix_dom_send", "intent": "book Studio B for Thursday"},
        {"interaction_item_id": "held_IqMDuvgWD", "intent": ""},
    ]
    notes = _rebuild(tmp_path, "fin_08", "autonomy_level", events, derived).blocks["PRIVATE_NOTES"]
    first = notes.split("\n\n")[0].split("\n")
    assert first[-3:] == [
        "The assistant is holding these items until you say what to do with them:",
        "- book Studio B for Thursday",
        "- an item (no description given)",
    ]
    assert "ix_dom_send" not in notes and "held_IqMDuvgWD" not in notes
    assert "Answer them in <message>" not in notes


def test_suppressed_outreach_gives_no_turn_no_notes_and_a_skipped_opening(tmp_path: Path) -> None:
    events, derived = _outreach_probe(delivered=False)
    script = _script("fin_09")
    send_goal, suppress_goal = (beat.goal for beat in script.beats)
    rebuilt = _rebuild(tmp_path, "fin_09", "proactive_outreach", events, derived)
    blocks = rebuilt.blocks

    assert blocks["CONVERSATION"] == "turn 0 — me:\nI saw it.\n\nturn 1 — assistant:\nNoted."
    assert blocks["PRIVATE_NOTES"].split("\n")[0] == "private notes after turn 1:"
    assert blocks["PRIVATE_NOTES"].count("private notes after turn") == 1
    assert OUTREACH_ACT not in blocks["PRIVATE_NOTES"] and "None" not in blocks["PRIVATE_NOTES"]
    assert f"1. {send_goal} (skipped)\n2. {suppress_goal} (now)" in blocks["SESSION_SETUP"]
    assert "(done)" not in blocks["SESSION_SETUP"]
    assert TRIGGERED_OPENING_NOTE in blocks["SESSION_SETUP"]


def test_delivered_outreach_is_turn_0_with_its_notes_and_no_step_to_come(tmp_path: Path) -> None:
    events, derived = _outreach_probe(delivered=True)
    blocks = _rebuild(tmp_path, "fin_09", "proactive_outreach", events, derived).blocks

    assert blocks["CONVERSATION"].startswith(
        "turn 0 — assistant:\nGwen emailed about the showcase.\n\nturn 1 — me:"
    )
    notes = blocks["PRIVATE_NOTES"].split("\n\n")
    assert [n.split("\n")[0] for n in notes] == ["private notes after turn 0:", "private notes after turn 2:"]
    assert f"- Changes it carried out in that reply: {OUTREACH_ACT}" in notes[0]
    assert f"- Changes it had already carried out earlier in this conversation: {OUTREACH_ACT}" in notes[1]
    assert "more step" not in blocks["SESSION_SETUP"]


def test_suppressed_outreach_holding_items_is_refused(tmp_path: Path) -> None:
    events, derived = _outreach_probe(delivered=False)
    derived[2]["held_items"] = [{"interaction_item_id": "x", "intent": "y"}]
    with pytest.raises(RebuildError, match="suppressed outreach holds items"):
        _rebuild(tmp_path, "fin_09", "proactive_outreach", events, derived)


def test_the_recorded_overlay_sets_the_sheet_and_the_who_row(tmp_path: Path) -> None:
    events = [_enter(1, "ask"), _me(2, "ask", "Which one?"), _pa(3, "ask", "This one.")]
    base = PreferenceMatrix.from_yaml(PREFERENCES)
    context = _script("fin_11").context
    own = base.matrix[context]["solution_breadth"].setting
    other = next(s for s in load_prompt_files()[1]["solution_breadth"]["who"] if s != own)
    patch = {"setting": other, "source": "forced_event", "confidence": "High"}
    overlay = {context: {"solution_breadth": patch}}

    flipped = _rebuild(tmp_path / "a", "fin_11", "solution_breadth", events, _derived(3), overlay)
    plain = _rebuild(tmp_path / "b", "fin_11", "solution_breadth", events, _derived(3))

    assert flipped.setting == other and plain.setting == own
    assert f"## Skill: solution_breadth = {other}\n" in flipped.blocks["ACTING_SHEET"]
    who = load_prompt_files()[1]["solution_breadth"]["who"]
    assert who[other] in _prompt(flipped, "solution_breadth").split("# QUESTION")[1]
    assert who[own] in _prompt(plain, "solution_breadth").split("# QUESTION")[1]


def test_a_two_beat_probe_gives_each_beats_first_turn_and_the_last_outline(tmp_path: Path) -> None:
    events = [
        _enter(1, "save_destination"), _me(2, "save_destination", "Save it to my notes."),
        _pa(3, "save_destination", "Where?"),
        _enter(4, "ask"), _me(5, "ask", "The server note."), _pa(6, "ask", "Done."),
    ]
    script = _script("pre_02")
    rebuilt = _rebuild(tmp_path, "pre_02", "process_visibility", events, _derived(3, 6))
    setup = rebuilt.blocks["SESSION_SETUP"]
    first, second = setup.index("Beat id: save_destination"), setup.index("Beat id: ask")
    ask_line = "This is the step where you ask for what this conversation is about."

    assert setup.index("(now)") < first < second
    assert f"1. {script.beats[0].goal} (done)\n2. {script.beats[1].goal} (now)" in setup
    assert ask_line not in setup[first:second] and ask_line in setup[second:]
    assert WITHHELD_FACTS_LINE not in setup
    assert rebuilt.blocks["ACTING_SHEET"].count("## Skill: ") == 1
    assert rebuilt.blocks["LIFE_CONTEXT"].startswith("- Now: Tuesday, 2026-02-03 09:30\n")


def test_conversation_turns_number_the_live_history(tmp_path: Path) -> None:
    events, derived = _autonomy_probe()
    rebuilt = _rebuild(tmp_path, "fin_08", "autonomy_level", events, derived)

    assert [(h["role"], h["t"]) for h in rebuilt.history] == [
        ("assistant", 2), ("user", 3), ("assistant", 4), ("user", 5)
    ]
    assert rebuilt.blocks["CONVERSATION"].split("\n\n")[1] == (
        "turn 1 — assistant:\nI can book it. Shall I go ahead?"
    )


@pytest.mark.parametrize("kind", ["beat_skipped", "skill_triggered"])
def test_an_event_the_rebuild_does_not_replay_is_refused(tmp_path: Path, kind: str) -> None:
    events, derived = _autonomy_probe()
    events.insert(3, {"kind": kind, "t": 99, "beat": "ask"})
    with pytest.raises(RebuildError, match=kind):
        _rebuild(tmp_path, "fin_08", "autonomy_level", events, derived)


def test_a_skill_triggered_after_the_actors_last_turn_changes_nothing(tmp_path: Path) -> None:
    """Reference rounds: the PA's last reply put capability_boundary in play,
    after the actor's last turn. Before that turn it is still refused (above)."""
    events, derived = _autonomy_probe()
    plain = _rebuild(tmp_path / "a", "fin_08", "autonomy_level", events, derived)
    events.insert(5, {"kind": "skill_triggered", "t": 99, "attribute": "capability_boundary",
                      "source": "triggered_skills"})
    trailing = _rebuild(tmp_path / "b", "fin_08", "autonomy_level", events, derived)

    assert trailing.blocks == plain.blocks and trailing.history == plain.history


def test_a_pa_turn_without_harness_keys_is_refused(tmp_path: Path) -> None:
    events, derived = _autonomy_probe()
    del derived[5]
    with pytest.raises(RebuildError, match="t=5"):
        _rebuild(tmp_path, "fin_08", "autonomy_level", events, derived)


# ----------------------------------------------------------------- parser


def test_a_valid_block_parses() -> None:
    parsed = parse_report(VALID)
    assert (parsed.score, parsed.reason, parsed.findings, parsed.outside_text) == (
        4, "In turn 1 it said 'Done' and that suited me.", (), False
    )


def test_a_quoted_reason_with_a_colon_loses_only_its_quotes() -> None:
    raw = (
        "<dimension_self_report>\nscore: 5\n"
        "reason: \"Turn 1 said 'Most stalls (confidence: medium)' and that helped.\"\n"
        "</dimension_self_report>"
    )
    assert parse_report(raw).reason == "Turn 1 said 'Most stalls (confidence: medium)' and that helped."


def test_text_outside_the_block_is_recorded_not_refused() -> None:
    parsed = parse_report("Here you go:\n" + VALID)
    assert parsed.score == 4 and parsed.outside_text is True


@pytest.mark.parametrize(
    ("raw", "finding"),
    [
        ("score: 4\nreason: fine", "dimension_self_report_missing"),
        (VALID + "\n" + VALID, "dimension_self_report_repeated"),
        (VALID.replace("score: 4", "score: 6"), "score_not_1_to_5"),
        (VALID.replace("score: 4", "score: 4.0"), "score_not_1_to_5"),
        (VALID.replace("score: 4", "score: true"), "score_not_1_to_5"),
        (VALID.replace("score: 4", "score: '4'"), "score_not_1_to_5"),
        (VALID.replace("that suited me.", "that suited me.\nIt was quick."), "dimension_self_report_malformed"),
        (VALID.replace("score: 4\n", "score: 4\nsetting: Low\n"), "dimension_self_report_malformed"),
        (VALID.replace("score: 4\nreason: In turn 1 it said 'Done' and that suited me.",
                       "reason: In turn 1 it said 'Done'.\nscore: 4"), "dimension_self_report_malformed"),
        (VALID.replace("In turn 1 it said 'Done' and that suited me.", '""'), "reason_empty"),
    ],
)
def test_a_block_off_contract_is_a_finding(raw: str, finding: str) -> None:
    parsed = parse_report(raw)
    assert parsed.score is None and finding in parsed.findings


# ----------------------------------------------------------------- caller


class _Provider(LLMProvider):
    """Scripted; reports a cost the way OpenAICompatProvider does, through
    `_extract_cost` on each answered call."""

    def __init__(self, responses: list[Any], cost: float | None = 0.01) -> None:
        super().__init__()
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []
        self.cost = cost

    def get_default_model(self) -> str:
        return "default"

    @classmethod
    def _extract_cost(cls, provider_response: Any) -> Any:
        return provider_response

    async def chat(self, messages, tools=None, model=None, **kwargs):
        self.calls.append({"messages": messages, "tools": tools, "model": model, **kwargs})
        response = self.responses.pop(0)
        if callable(response):
            return await response()
        response = response if isinstance(response, LLMResponse) else LLMResponse(content=response)
        if response.finish_reason != "error":
            self._extract_cost(self.cost)
        return response


def _reporter(provider, *, retries=5):
    return DimensionSelfReporter(provider, model="google/gemini-3.8-flash", output_contract_retries=retries)


@pytest.fixture
def instant_sleep(monkeypatch):
    async def sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr(asyncio, "sleep", sleep)


def test_the_call_uses_the_session_end_self_report_params() -> None:
    provider = _Provider([VALID])
    result = asyncio.run(_reporter(provider).generate("THE PROMPT"))

    assert result["status"] == "ok" and (result["score"], result["reason"]) == (4, parse_report(VALID).reason)
    call = provider.calls[0]
    assert call["messages"] == [{"role": "system", "content": "THE PROMPT"}]
    assert call["tools"] == [] and call["model"] == "google/gemini-3.8-flash"
    assert call["max_tokens"] == SELF_REPORT_MAX_TOKENS
    assert call["reasoning_effort"] is None


def test_an_invalid_attempt_is_re_asked_with_the_self_report_correction() -> None:
    provider = _Provider(["no block here", VALID])
    result = asyncio.run(_reporter(provider).generate("P"))

    assert result["status"] == "ok" and len(provider.calls) == 2
    assert provider.calls[1]["messages"][1:] == [
        {"role": "assistant", "content": "no block here"},
        {"role": "user", "content": (
            "The output violated the private schema: dimension_self_report_missing. "
            "Regenerate the same result in the required blocks without discussing this retry."
        )},
    ]
    assert [c["findings"] for c in result["calls"]] == [["dimension_self_report_missing"], []]


def test_six_invalid_attempts_exhaust_the_contract_and_keep_every_raw() -> None:
    provider = _Provider([f"bad {i}" for i in range(6)])
    result = asyncio.run(_reporter(provider).generate("P"))

    assert result["status"] == "contract_exhausted" and result["score"] is None
    assert [c["raw"] for c in result["calls"]] == [f"bad {i}" for i in range(6)]
    assert [c["attempt"] for c in result["calls"]] == [1, 2, 3, 4, 5, 6]


def test_an_empty_completion_is_retried_and_only_empties_end_as_empty_output(instant_sleep) -> None:
    provider = _Provider([LLMResponse(content=None, finish_reason="length"), VALID])
    assert asyncio.run(_reporter(provider).generate("P"))["status"] == "ok"
    assert len(provider.calls) == 2

    provider = _Provider([LLMResponse(content="", finish_reason="length")] * 5)
    result = asyncio.run(_reporter(provider).generate("P"))
    assert result["status"] == "empty_output" and len(result["calls"]) == 5


def test_an_error_response_is_a_provider_error_and_never_parsed() -> None:
    provider = _Provider([LLMResponse(content="Error calling LLM: boom", finish_reason="error")] * 4)
    result = asyncio.run(_reporter(provider).generate("P"))

    assert result["status"] == "provider_error"
    assert len(result["calls"]) == 1 and result["calls"][0]["findings"] == []
    assert result["cost_missing_calls"] == 1


def test_the_wall_clock_deadline_is_a_provider_error(monkeypatch) -> None:
    async def stall():
        await asyncio.Event().wait()

    monkeypatch.setattr(dsr, "CALL_DEADLINE_S", 0.01)
    result = asyncio.run(_reporter(_Provider([stall])).generate("P"))
    assert result["status"] == "provider_error" and result["calls"][0]["finish_reason"] == "timeout"


def test_a_call_without_a_reported_cost_is_counted_as_missing() -> None:
    result = asyncio.run(_reporter(_Provider([VALID], cost=None)).generate("P"))

    assert result["calls"][0]["cost_usd"] is None and result["cost_missing_calls"] == 1
    assert result["cost_usd_total"] == 0


def test_the_real_provider_reports_usage_and_cost_per_call(monkeypatch) -> None:
    provider = OpenAICompatProvider(api_key="test-key", api_base="http://127.0.0.1:9")

    async def create(**kwargs):
        return {
            "choices": [{"message": {"content": VALID}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 120, "completion_tokens": 30, "total_tokens": 150, "cost": 0.0021},
        }

    monkeypatch.setattr(provider._client.chat.completions, "create", create)
    result = asyncio.run(_reporter(provider).generate("P"))

    assert result["status"] == "ok" and result["calls"][0]["cost_usd"] == pytest.approx(0.0021)
    assert result["usage_total"] == {"prompt_tokens": 120, "completion_tokens": 30}
    assert result["cost_usd_total"] == pytest.approx(0.0021)


def test_one_provider_cannot_meter_two_probes() -> None:
    provider = _Provider([VALID])
    _reporter(provider)
    with pytest.raises(ValueError, match="one provider per probe"):
        _reporter(provider)
