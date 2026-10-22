"""The mechanical guards: fence strip, schema, evidence, retries, budget."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from simulator.judge.guards import (
    ASK_GATE_VALUES,
    JudgeBudgetExhausted,
    JudgeEvidenceError,
    JudgeEvidenceInvalid,
    JudgeSchemaError,
    JudgeTimeout,
    JudgeUnparseable,
    MAX_NUM_PREDICT,
    TurnBudget,
    VERDICT_VALUES,
    Verdict,
    YES_NO_VALUES,
    ask,
    check_evidence,
    normalise,
    number_sentences,
    numbered_block,
    parse_verdicts,
    strip_fences,
)

FIXTURE = Path(__file__).parent / "fixture"

#: The quoting side of the two questions these tests stand a reply on, and the
#: blocks each may point at.
ASK_QUOTES_ON = {"carries_an_ask": ("ask",)}
ASK_BLOCKS = {"carries_an_ask": ("candidate_turn",)}
WITHHELD_QUOTES_ON = {"withheld_fact_before_ask": ("yes",)}

GOOD = json.dumps(
    {
        "verdicts": [
            {
                "question_id": "carries_an_ask",
                "verdict": "ask",
                "evidence": "candidate_turn#1",
                "reason": "an instruction to the assistant",
            }
        ]
    }
)
CANDIDATE = "Update the year field in the Sarpong entry to 2021."
BLOCKS = {"candidate_turn": CANDIDATE}


def _resolve(verdicts, blocks=None, quotes_on=None, points_at=None):
    """The guard's own resolution, on one question's answers."""
    return check_evidence(
        verdicts,
        blocks if blocks is not None else BLOCKS,
        quotes_on or ASK_QUOTES_ON,
        points_at or ASK_BLOCKS,
    )


class FakeClient:
    """Replays a queue of raw replies and counts the requests."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0
        self.prompts: list[str] = []

    def generate(self, **kwargs):
        self.calls += 1
        self.prompts.append(kwargs["prompt"])
        raw = self.replies.pop(0) if self.replies else ""

        class Call:
            def __init__(self, raw):
                self.raw = raw
                self.seconds = 0.1
                self.prompt_tokens = 10
                self.eval_tokens = 5

        return Call(raw)


# ------------------------------------------------------------- fence strip


@pytest.mark.parametrize(
    "raw",
    [
        f"```json\n{GOOD}\n```",
        f"```\n{GOOD}\n```",
        f"Here is my answer.\n{GOOD}",
        f"{GOOD}\n\nI hope this helps.",
        f"Sure!\n```json\n{GOOD}\n```\nLet me know.",
    ],
)
def test_strip_fences_recovers_the_object(raw):
    assert json.loads(strip_fences(raw))["verdicts"][0]["verdict"] == "ask"


def test_strip_fences_leaves_a_clean_object_alone():
    assert strip_fences(GOOD) == GOOD


def test_recorded_raw_replies_are_not_fence_pollution():
    """The 15.2% of pass-1 replies that never parsed are not a fence shape.

    They are the two-character `{"` and the empty string of the stream=false
    transport bug, so the fence strip cannot recover them and does not claim to.
    The three fixture rows that ARE fence-shaped are recovered.
    """
    rows = [
        json.loads(line)
        for line in (FIXTURE / "corpus" / "recorded_raw.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    recorded = [row for row in rows if row["source"] != "fixture"]
    synthetic = [row for row in rows if row["source"] == "fixture"]

    def parses(text):
        try:
            json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return False
        return True

    assert recorded and not any(parses(strip_fences(row["raw"])) for row in recorded)
    assert synthetic and all(parses(strip_fences(row["raw"])) for row in synthetic)


# ------------------------------------------------------------------- schema


def test_parse_verdicts_reads_the_schema():
    verdicts = parse_verdicts(GOOD, allowed={"carries_an_ask": ASK_GATE_VALUES})
    assert [v.question_id for v in verdicts] == ["carries_an_ask"]
    assert verdicts[0].is_yes


def test_ask_gate_rejects_obeyed_and_violated():
    raw = GOOD.replace('"ask"', '"violated"')
    with pytest.raises(JudgeSchemaError):
        parse_verdicts(raw, allowed={"carries_an_ask": ASK_GATE_VALUES})


def test_question_ids_must_match_exactly():
    with pytest.raises(JudgeSchemaError):
        parse_verdicts(GOOD, allowed={"carries_an_ask": ASK_GATE_VALUES, "extra": ASK_GATE_VALUES})


def test_an_empty_reason_is_a_schema_failure():
    """Regress23 finding 2: ~8 live answers came back with `evidence` and
    `reason` both empty. Schema-legal, and a replay cannot tell a considered
    `no` from a dropped one."""
    raw = GOOD.replace('"an instruction to the assistant"', '""')
    with pytest.raises(JudgeSchemaError):
        parse_verdicts(raw, allowed={"carries_an_ask": ASK_GATE_VALUES})


def test_a_whitespace_only_reason_is_a_schema_failure():
    """The schema's `minLength` stops "" at the sampler but not " " or "\n"."""
    for blank in (" ", "\n", " \t "):
        raw = GOOD.replace('"an instruction to the assistant"', f'"{blank}"')
        with pytest.raises(JudgeSchemaError):
            parse_verdicts(raw, allowed={"carries_an_ask": ASK_GATE_VALUES})


def test_empty_evidence_on_a_quoting_answer_is_an_evidence_failure():
    """`ask` asserts the request is there, so it has to point at it."""
    raw = GOOD.replace('"candidate_turn#1"', '""')
    verdicts = parse_verdicts(raw, allowed={"carries_an_ask": ASK_GATE_VALUES})
    with pytest.raises(JudgeEvidenceError):
        _resolve(verdicts)


def test_evidence_is_not_checked_on_an_answer_that_asserts_an_absence():
    """The owner's ruling of 2026-09-09: 没违反不引用. `not_applicable` says the
    turn puts nothing to the assistant, so whatever the model wrote in
    `evidence` — a reason, a reference, nothing at all — is not read."""
    for written in ("", "the availability question is all he asks for here."):
        verdicts = [Verdict("carries_an_ask", "not_applicable", written, "r")]
        assert _resolve(verdicts)[0].evidence == written


def test_a_verdict_question_may_point_on_obeyed_instead():
    """gate_cue_missing inverts the default: `obeyed` points at the sentence
    that carries the cue, `violated` says the meaning is nowhere in the turn."""
    quotes_on = {"gate_cue_missing": ("obeyed",)}
    points_at = {"gate_cue_missing": ("candidate_turn",)}
    kwargs = {"quotes_on": quotes_on, "points_at": points_at}
    _resolve(
        [Verdict("gate_cue_missing", "violated", "the cue is nowhere here", "r")],
        **kwargs,
    )
    with pytest.raises(JudgeEvidenceError):
        _resolve(
            [Verdict("gate_cue_missing", "obeyed", "not a reference at all", "r")],
            **kwargs,
        )


def test_not_applicable_may_leave_evidence_empty():
    raw = json.dumps(
        {
            "verdicts": [
                {
                    "question_id": "carries_an_ask",
                    "verdict": "not_applicable",
                    "evidence": "",
                    "reason": "the turn only reports",
                }
            ]
        }
    )
    verdicts = parse_verdicts(raw, allowed={"carries_an_ask": ASK_GATE_VALUES})
    assert not verdicts[0].is_yes


# ----------------------------------------------------------------- evidence


def test_evidence_normalisation_folds_case_whitespace_and_curly_quotes():
    assert normalise("  The  “Year”  Field ") == 'the "year" field'


# ---------------------------------------------------------------- numbering

TWO_LINES = "First line here. Second sentence!\n- a bullet row\n\nLast one?"


def test_sentences_are_numbered_by_line_then_by_terminator():
    """A markdown bullet or a table row is a line, so it is a piece of its own
    whatever it ends with; inside a line the cut is a terminator and a space."""
    assert number_sentences(TWO_LINES) == [
        "First line here.",
        "Second sentence!",
        "- a bullet row",
        "Last one?",
    ]


def test_the_numbering_keeps_each_piece_verbatim():
    assert number_sentences("  Dr. Sarpong replied.   Then left. ") == [
        "Dr.",
        "Sarpong replied.",
        "Then left.",
    ]


def test_a_block_is_printed_with_those_numbers():
    assert numbered_block("candidate_turn", "One. Two.") == (
        "<candidate_turn>\n[1] One.\n[2] Two.\n</candidate_turn>"
    )


# ----------------------------------------------------------------- references


def test_a_reference_resolves_to_the_sentence_it_names():
    verdicts = parse_verdicts(GOOD, allowed={"carries_an_ask": ASK_GATE_VALUES})
    (resolved,) = _resolve(verdicts)
    assert resolved.evidence == CANDIDATE
    assert resolved.evidence_ref == "candidate_turn#1"


def test_several_references_resolve_in_the_order_given():
    """Every spelling the format allows: the block named once, named twice, and
    named once with its sentence numbers listed after it — the shape about 1% of
    this model's answers write (owner ruling, 2026-09-16)."""
    block = {"candidate_turn": "Open the file. Then close it. Finally rest."}
    for written in (
        "candidate_turn#3,#1",
        "candidate_turn#3, candidate_turn#1",
        "candidate_turn#3,1",
        "candidate_turn#3, 1",
    ):
        verdict = Verdict("carries_an_ask", "ask", written, "r")
        (resolved,) = _resolve([verdict], blocks=block)
        assert resolved.evidence == "Finally rest. Open the file."
        assert resolved.evidence_ref == written


def test_a_block_the_question_does_not_point_at_is_refused():
    verdict = Verdict("carries_an_ask", "ask", "pa_previous_turn#1", "r")
    with pytest.raises(JudgeEvidenceError, match="not a block this question"):
        _resolve([verdict], blocks={**BLOCKS, "pa_previous_turn": "Anything else?"})


def test_a_sentence_number_that_is_not_there_is_refused():
    verdict = Verdict("carries_an_ask", "ask", "candidate_turn#4", "r")
    with pytest.raises(JudgeEvidenceError, match=r"no sentence \[4\]"):
        _resolve([verdict])


def test_one_number_of_a_listed_pair_that_is_not_there_is_refused():
    """A pointer listing several numbers is resolved whole: one that is not
    there refuses the answer, as it does when it is written on its own."""
    verdict = Verdict("carries_an_ask", "ask", "candidate_turn#1,4", "r")
    with pytest.raises(JudgeEvidenceError, match=r"no sentence \[4\]"):
        _resolve([verdict])


def test_a_quote_where_a_reference_belongs_is_refused():
    """The failure the ruling of 2026-09-09 is about: the model copied words.
    Copied words are not a reference, whether or not they are in the block."""
    for written in ("Update the year field", "Update the year column", "#0"):
        verdict = Verdict("carries_an_ask", "ask", written, "r")
        with pytest.raises(JudgeEvidenceError):
            _resolve([verdict])


# ------------------------------------------------------------------ retries


def _ask(client, blocks=None):
    return ask(
        client,
        system="s",
        prompt="p",
        allowed={"carries_an_ask": ASK_GATE_VALUES},
        blocks_by_name=blocks or BLOCKS,
        quotes_on=ASK_QUOTES_ON,
        question_blocks=ASK_BLOCKS,
    )


def test_a_parse_failure_is_re_asked_twice_then_raises():
    client = FakeClient(["not json", "still not json", "nope"])
    with pytest.raises(JudgeUnparseable) as excinfo:
        _ask(client)
    assert client.calls == 3
    assert excinfo.value.event["event"] == "judge_unparseable"
    assert excinfo.value.event["attempts"] == 3


def test_a_parse_failure_that_recovers_returns_the_verdicts():
    client = FakeClient(["```json\nbroken", GOOD])
    verdicts = _ask(client)
    assert client.calls == 2
    assert verdicts[0].verdict == "ask"


class CuttingClient:
    """Replays (raw, was_cut) pairs and records the width each request asked
    for, the way the real client reports `done_reason == "length"`."""

    def __init__(self, replies, base_width: int = 600):
        self.replies = list(replies)
        self.base_width = base_width
        self.calls = 0
        self.widths: list[int] = []

    def generate(self, *, system, prompt, allowed, num_predict_kind="yes_no", num_predict=None):
        self.calls += 1
        raw, cut = self.replies.pop(0)
        width = num_predict if num_predict is not None else self.base_width
        self.widths.append(width)

        class Call:
            def __init__(self, raw, cut, width):
                self.raw = raw
                self.seconds = 0.1
                self.prompt_tokens = 1748
                self.eval_tokens = width if cut else 120
                self.num_predict = width
                self.truncated = cut

        return Call(raw, cut, width)


def _ask_wide(client):
    return ask(
        client,
        system="s",
        prompt="p",
        allowed={"carries_an_ask": ASK_GATE_VALUES},
        blocks_by_name=BLOCKS,
        quotes_on=ASK_QUOTES_ON,
        question_blocks=ASK_BLOCKS,
        num_predict_kind="wide",
    )


def test_a_cut_reply_is_re_asked_once_at_twice_the_width():
    """Well-formed JSON that stops mid-value cannot be fixed by sending the
    identical request again; the smoke run of 2026-09-09 spent three 30-second
    calls proving it."""
    client = CuttingClient([('{"verdicts": [{"verdict": "yes', True), (GOOD, False)])
    verdicts = _ask_wide(client)
    assert client.calls == 2
    assert client.widths == [600, 1200]
    assert verdicts[0].verdict == "ask"


def test_a_second_cut_raises_at_once_and_names_the_cause():
    cut = '{"verdicts": [{"verdict": "yes'
    client = CuttingClient([(cut, True), (cut, True)])
    with pytest.raises(JudgeUnparseable) as excinfo:
        _ask_wide(client)
    assert client.calls == 2  # never a third identical attempt
    event = excinfo.value.event
    assert event["truncated"] is True
    assert event["completion_tokens"] == [600, 1200]
    assert event["attempts"] == 2


def test_the_widened_width_is_capped():
    """A per-fact group already asks for more than the general tier, so the
    doubling has a ceiling of its own."""
    cut = '{"verdicts": [{"verdict": "yes'
    client = CuttingClient([(cut, True), (cut, True)], base_width=2368)
    with pytest.raises(JudgeUnparseable):
        _ask_wide(client)
    assert client.widths == [2368, MAX_NUM_PREDICT]


def test_a_parse_failure_that_is_not_a_cut_still_gets_its_two_re_asks():
    client = CuttingClient([("not json", False), ("still not", False), (GOOD, False)])
    verdicts = _ask_wide(client)
    assert client.calls == 3
    assert client.widths == [600, 600, 600]
    assert verdicts[0].verdict == "ask"


def test_invalid_evidence_is_re_asked_once_then_raises():
    bad = GOOD.replace('"candidate_turn#1"', '"candidate_turn#7"')
    client = FakeClient([bad, bad])
    with pytest.raises(JudgeEvidenceInvalid) as excinfo:
        _ask(client)
    assert client.calls == 2
    event = excinfo.value.event
    assert event["event"] == "judge_evidence_invalid"
    # The turn the judge read is dropped with the attempt, so the event is the
    # only place a read can hold the reference against the numbered block the
    # judge was given, and against its own reply.
    assert event["block_id"] == "candidate_turn"
    assert event["block_text"] == (
        f"<candidate_turn>\n[1] {CANDIDATE}\n</candidate_turn>"
    )
    assert event["evidence"] == "candidate_turn#7"
    assert event["raw_response"] == bad
    # The re-ask is not the identical request: in the live batch of 2026-09-09
    # every blind re-send came back with the same wrong answer, so this one says
    # what was wrong with the last one (owner ruling, contract §4).
    first, second = client.prompts
    assert first == "p"
    assert second.startswith("p")
    assert "carries_an_ask" in second
    assert "candidate_turn#7" in second
    assert "has no sentence [7]" in second


def test_evidence_is_checked_against_the_block_the_question_points_at():
    raw = json.dumps(
        {
            "verdicts": [
                {
                    "question_id": "withheld_fact_before_ask",
                    "verdict": "yes",
                    "evidence": "assistant_turns_so_far#1",
                    "reason": "focus, the assistant asked for it",
                }
            ]
        }
    )
    verdicts = parse_verdicts(raw, allowed={"withheld_fact_before_ask": YES_NO_VALUES})
    blocks = {
        "candidate_turn": CANDIDATE,
        "assistant_turns_so_far": "What should the review focus on? Tell me.",
    }
    # the question that may read only the candidate turn refuses the reference
    with pytest.raises(JudgeEvidenceError):
        check_evidence(
            verdicts,
            blocks,
            WITHHELD_QUOTES_ON,
            {"withheld_fact_before_ask": ("candidate_turn",)},
        )
    (resolved,) = check_evidence(
        verdicts,
        blocks,
        WITHHELD_QUOTES_ON,
        {"withheld_fact_before_ask": ("assistant_turns_so_far",)},
    )
    assert resolved.evidence == "What should the review focus on?"


# ------------------------------------------------------------------- budget


def test_budget_counts_calls_not_retries():
    budget = TurnBudget("turn-1", limit=2)
    client = FakeClient(["not json", GOOD, GOOD])
    for _ in range(2):
        ask(
            client,
            system="s",
            prompt="p",
            allowed={"carries_an_ask": ASK_GATE_VALUES},
            blocks_by_name=BLOCKS,
            quotes_on=ASK_QUOTES_ON,
            question_blocks=ASK_BLOCKS,
            budget=budget,
        )
    assert client.calls == 3  # one re-ask, two groups
    assert budget.used == 2


def test_budget_exhausted_carries_the_turn_uid():
    budget = TurnBudget("turn-1", limit=1)
    budget.spend()
    with pytest.raises(JudgeBudgetExhausted) as excinfo:
        budget.spend()
    assert excinfo.value.event == {
        "event": "judge_budget_exhausted",
        "turn_uid": "turn-1",
        "used": 2,
    }


def test_budget_default_reads_the_env_var(monkeypatch):
    monkeypatch.setenv("MEMPA_JUDGE_TURN_BUDGET", "3")
    assert TurnBudget("turn-1").limit == 3
    monkeypatch.delenv("MEMPA_JUDGE_TURN_BUDGET")
    assert TurnBudget("turn-1").limit == 14


def test_every_failure_event_carries_its_dict():
    events = [
        JudgeUnparseable(question_id="q", attempts=3, raw_response="x").event,
        JudgeEvidenceInvalid(question_id="q", evidence="x").event,
        JudgeTimeout(question_ids=["q"], seconds=15.0).event,
        JudgeBudgetExhausted(turn_uid="t", used=13).event,
    ]
    assert [event["event"] for event in events] == [
        "judge_unparseable",
        "judge_evidence_invalid",
        "judge_timeout",
        "judge_budget_exhausted",
    ]


# ------------------------------------------------------------- empty groups


def test_no_questions_means_no_call():
    """An empty group is not a question group, so §2 has no call to make — and
    an empty `format` enum is not something Ollama can sample against."""
    client = FakeClient([GOOD])
    budget = TurnBudget("turn-1", limit=1)
    assert ask(
        client,
        system="s",
        prompt="p",
        allowed={},
        blocks_by_name={},
        quotes_on={},
        question_blocks={},
        budget=budget,
    ) == []
    assert client.calls == 0
    assert budget.used == 0


def test_the_schema_forbids_an_empty_reason():
    """Regress23 finding 2, at the sampler: constrained decoding honours
    `minLength`, so "" cannot come back at all. `verdict_schema` rebuilds the
    item properties per call and must carry the constraint through."""
    from simulator.judge.client import VERDICT_SCHEMA, verdict_schema

    item = VERDICT_SCHEMA["properties"]["verdicts"]["items"]["properties"]
    assert item["reason"] == {"type": "string", "minLength": 1}

    built = verdict_schema({"carries_an_ask": ASK_GATE_VALUES})
    built_item = built["properties"]["verdicts"]["items"]["properties"]
    assert built_item["reason"] == {"type": "string", "minLength": 1}


def test_the_client_refuses_to_build_a_schema_with_no_questions():
    from simulator.judge.client import verdict_schema

    with pytest.raises(ValueError, match="at least one question"):
        verdict_schema({})


# --------------------------------------------------- accented-name folding


def test_normalise_folds_an_accent_to_its_base_letter():
    """`normalise` outlived the substring check it was written for: the
    withheld-fact compose reads the named noun out of the resolved sentence
    with it."""
    assert normalise("Brandão") == normalise("Brandào") == "brandao"


def test_a_respelled_name_costs_nothing_now_the_evidence_is_a_number():
    """Three of the eight evidence failures of 2026-09-09 were the model
    re-spelling à for ã in an otherwise verbatim quote. A reference has no
    letters to re-spell, and the sentence comes back off the block itself."""
    block = {"candidate_turn": "Brandão is the foundational source."}
    verdict = Verdict("carries_an_ask", "ask", "candidate_turn#1", "r")
    assert _resolve([verdict], blocks=block)[0].evidence == (
        "Brandão is the foundational source."
    )


def test_a_blank_reason_is_re_asked_with_feedback_naming_the_question():
    """Decoding is deterministic, so the identical re-ask returns the identical
    blank `reason` (regress23m/acc_087 died on three of them). The second
    request must carry feedback, and a good answer to it must stand."""
    blank = json.dumps(
        {
            "verdicts": [
                {
                    "question_id": "carries_an_ask",
                    "verdict": "ask",
                    "evidence": "candidate_turn#1",
                    "reason": " ",
                }
            ]
        }
    )
    client = FakeClient([blank, GOOD])
    verdicts = _ask(client)
    assert client.calls == 2
    assert verdicts[0].verdict == "ask"
    assert client.prompts[0] == "p"
    assert "carries_an_ask: reason is empty" in client.prompts[1]
    assert "`reason`" in client.prompts[1]
