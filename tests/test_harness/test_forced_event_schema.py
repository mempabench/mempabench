from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from harness.forced_event_schema import (
    ForcedEventTranscriptLoadError,
    load_forced_event_transcript,
    validate_forced_event_transcript,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRANSCRIPT_PATH = (
    PROJECT_ROOT
    / "Authoring/sessions/forced/transcripts/user_a/evolv_05/transcript.yaml"
)
ALL_TRANSCRIPT_PATHS = sorted(
    (PROJECT_ROOT / "Authoring/sessions/forced/transcripts/user_a").glob(
        "*/transcript.yaml"
    )
)

# The hand-authored exemplar. It carries shapes a generated transcript is free
# not to author -- in particular an `environment_action` -- so the checks that
# need one present to mutate read it instead of whichever transcript the corpus
# currently holds. It deliberately does NOT bind to the corpus: its
# `source_event_path` names a fixture draft, so it is only ever loaded through
# `load_forced_event_transcript`, the layer that owns the shape rules. The
# repo-source binding is tested against the five landed transcripts above.
FIXTURE_TRANSCRIPT_PATH = (
    PROJECT_ROOT / "tests/fixtures/authoring/forced_event_transcript_evolv_05.yaml"
)


def _email_bearing_transcript() -> dict | None:
    """First committed transcript that authors an email send, if any exists."""
    for path in ALL_TRANSCRIPT_PATHS:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        for exchange in data["exchanges"]:
            for item in exchange["items"]:
                if item.get("name") == "mcp_state_email_message_send":
                    return data
    return None


def _valid_transcript() -> dict:
    return copy.deepcopy(yaml.safe_load(TRANSCRIPT_PATH.read_text(encoding="utf-8")))


def _fixture_transcript() -> dict:
    return copy.deepcopy(
        yaml.safe_load(FIXTURE_TRANSCRIPT_PATH.read_text(encoding="utf-8"))
    )


def _items_of_type(data: dict, item_type: str) -> list[dict]:
    return [
        item
        for exchange in data["exchanges"]
        for item in exchange["items"]
        if item["type"] == item_type
    ]


def _write_transcript(tmp_path: Path, data: dict) -> Path:
    path = tmp_path / "transcript.yaml"
    path.write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return path


def _validate(tmp_path: Path, data: dict):
    return validate_forced_event_transcript(
        _write_transcript(tmp_path, data),
        project_root=PROJECT_ROOT,
    )


def test_valid_transcript_matches_repo_sources(tmp_path: Path) -> None:
    transcript = _validate(tmp_path, _valid_transcript())

    assert transcript.event.event_id == "evolv_05"
    # How many exchanges the transcript runs for is the draft's business -- a
    # forced event runs for as many beats as its story needs, and
    # `validate_forced_event_transcript` is what checks the two agree. What
    # this test owns is that the landed file round-trips through the loader
    # and still binds to its repo sources.
    assert transcript.exchanges


@pytest.mark.parametrize(
    "transcript_path",
    ALL_TRANSCRIPT_PATHS,
    ids=[path.parent.name for path in ALL_TRANSCRIPT_PATHS],
)
def test_all_user_a_forced_transcripts_match_repo_sources(
    transcript_path: Path,
) -> None:
    transcript = validate_forced_event_transcript(
        transcript_path,
        project_root=PROJECT_ROOT,
    )

    assert transcript.event.event_id == transcript_path.parent.name


def test_transcript_can_author_email_body_but_not_frozen_attachment(
    tmp_path: Path,
) -> None:
    data = _email_bearing_transcript()
    if data is None:
        pytest.skip(
            "no committed forced transcript authors an email send in the pilot "
            "corpus; re-enable at full corpus regeneration"
        )
    send = next(
        item
        for exchange in data["exchanges"]
        for item in exchange["items"]
        if item.get("name") == "mcp_state_email_message_send"
    )
    send["arguments"]["body"] = "A different fixed collaborator message."

    validate_forced_event_transcript(
        _write_transcript(tmp_path, data),
        project_root=PROJECT_ROOT,
    )

    send["arguments"]["attachments"] = ["another_draft.txt"]
    with pytest.raises(ForcedEventTranscriptLoadError, match="arguments"):
        validate_forced_event_transcript(
            _write_transcript(tmp_path, data),
            project_root=PROJECT_ROOT,
        )


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("event_id", "evolv_04", "event_id"),
        ("persona_id", "user_b", "persona_id"),
        ("source_acc_num", 74, "source_acc_num"),
    ],
)
def test_repo_source_mismatch_is_rejected(
    tmp_path: Path,
    field: str,
    value: object,
    expected: str,
) -> None:
    data = _valid_transcript()
    data["event"][field] = value

    with pytest.raises(ForcedEventTranscriptLoadError, match=expected):
        _validate(tmp_path, data)


def test_post_setting_must_have_actor_skill_spec(tmp_path: Path) -> None:
    data = _valid_transcript()
    data["event"]["post_setting"] = "Conceal"

    with pytest.raises(ForcedEventTranscriptLoadError, match="post_setting"):
        _validate(tmp_path, data)


@pytest.mark.parametrize(
    ("problem", "expected"),
    [
        ("missing", r"call_id\s*\n?\s*Field required"),
        ("duplicate", "duplicate call_id values are not allowed"),
    ],
)
def test_tool_call_id_is_required_and_unique(
    tmp_path: Path,
    problem: str,
    expected: str,
) -> None:
    """The duplicate used to go into a new exchange of its own, where the
    visibility rule ("a tool call never stands alone") rejected the transcript
    before the uniqueness check ran -- and `match="call_id"` matched the
    `tmp_path` directory, which pytest names after this test, rather than the
    error. So the copy goes at the end of an exchange that already carries
    PA-visible text, keeping offsets increasing, and each case matches the
    message its own rule raises.
    """
    data = _valid_transcript()
    tool_call = _items_of_type(data, "pa_tool_call")[0]
    if problem == "missing":
        del tool_call["call_id"]
    else:
        duplicate = copy.deepcopy(tool_call)
        last = data["exchanges"][-1]
        duplicate["offset_seconds"] = (
            max(item["offset_seconds"] for item in last["items"]) + 1
        )
        last["items"].append(duplicate)

    with pytest.raises(ForcedEventTranscriptLoadError, match=expected):
        _validate(tmp_path, data)


def test_third_party_message_requires_visible_attribution(tmp_path: Path) -> None:
    data = _fixture_transcript()
    _items_of_type(data, "environment_action")[0]["content"] = (
        "Can your agent just tell you whether to bring the umbrella?"
    )

    with pytest.raises(ForcedEventTranscriptLoadError, match="attribution"):
        load_forced_event_transcript(_write_transcript(tmp_path, data))


def test_direct_third_party_message_can_be_attributed(tmp_path: Path) -> None:
    data = _fixture_transcript()
    action = _items_of_type(data, "environment_action")[0]
    exchange = next(
        exchange for exchange in data["exchanges"] if action in exchange["items"]
    )
    exchange["items"][exchange["items"].index(action)] = {
        "offset_seconds": action["offset_seconds"],
        "type": "message",
        "source": {"kind": "third_party", "id": "marcus_chen"},
        "visibility": "both",
        "content": "Marcus Chen: Please give us one departure time.",
    }
    data["assertions"]["expected_environment_actions"] = []

    transcript = load_forced_event_transcript(_write_transcript(tmp_path, data))

    source = next(
        item.source
        for exchange in transcript.exchanges
        for item in exchange.items
        if getattr(item, "type", None) == "message"
        and not isinstance(item.source, str)
    )
    assert source.id == "marcus_chen"


def test_pa_message_cannot_be_god_only(tmp_path: Path) -> None:
    data = _valid_transcript()
    pa_message = next(
        item
        for item in _items_of_type(data, "message")
        if item["source"] == "pa"
    )
    pa_message["visibility"] = "god_only"

    with pytest.raises(ForcedEventTranscriptLoadError, match="god_only"):
        _validate(tmp_path, data)


@pytest.mark.parametrize(
    "content",
    [
        "The preference shift is Express -> Hide.",
        "Authoring rationale: create social friction.",
    ],
)
def test_pa_visible_content_cannot_leak_authoring_metadata(
    tmp_path: Path,
    content: str,
) -> None:
    data = _valid_transcript()
    _items_of_type(data, "message")[0]["content"] = content

    with pytest.raises(ForcedEventTranscriptLoadError, match="authoring metadata"):
        _validate(tmp_path, data)


def test_offsets_must_increase_across_the_whole_event(tmp_path: Path) -> None:
    data = _valid_transcript()
    items = [item for exchange in data["exchanges"] for item in exchange["items"]]
    items[1]["offset_seconds"] = items[0]["offset_seconds"]

    with pytest.raises(ForcedEventTranscriptLoadError, match="strictly increasing"):
        _validate(tmp_path, data)


def test_source_event_path_cannot_escape_repo(tmp_path: Path) -> None:
    data = _valid_transcript()
    data["event"]["source_event_path"] = "../evolv_05.yaml"

    with pytest.raises(ForcedEventTranscriptLoadError, match="source_event_path"):
        _validate(tmp_path, data)


def test_unknown_fields_are_rejected(tmp_path: Path) -> None:
    data = _valid_transcript()
    data["event"]["post_seting"] = "Hide"

    with pytest.raises(ForcedEventTranscriptLoadError, match="post_seting"):
        _validate(tmp_path, data)


def test_fixture_precondition_may_name_any_existing_world_file(
    tmp_path: Path,
) -> None:
    data = _valid_transcript()
    data["preconditions"]["required_fixtures"][0]["path"] = "email/inbox.jsonl"

    transcript = _validate(tmp_path, data)

    assert transcript.preconditions.required_fixtures[0].path == "email/inbox.jsonl"


def test_fixture_precondition_source_must_exist(tmp_path: Path) -> None:
    data = _valid_transcript()
    data["preconditions"]["required_fixtures"][0]["path"] = "weather/missing.json"

    with pytest.raises(ForcedEventTranscriptLoadError, match="does not exist"):
        _validate(tmp_path, data)


def test_outline_fixture_cannot_be_omitted(tmp_path: Path) -> None:
    data = _valid_transcript()
    data["preconditions"]["required_fixtures"] = []

    with pytest.raises(ForcedEventTranscriptLoadError, match="required_fixtures"):
        _validate(tmp_path, data)


def _source_draft_pattern(data: dict) -> str:
    draft = yaml.safe_load(
        (PROJECT_ROOT / data["event"]["source_event_path"]).read_text(encoding="utf-8")
    )
    return draft["simulation"]["assistance_emergence"]["pattern"]


def test_transcript_pattern_must_match_source_draft(tmp_path: Path) -> None:
    data = _valid_transcript()
    actual = _source_draft_pattern(data)
    wrong = "partial_intent" if actual != "partial_intent" else "contextual_signal"
    data["event"]["assistance_emergence_pattern"] = wrong
    with pytest.raises(
        ForcedEventTranscriptLoadError,
        match="assistance_emergence_pattern does not match the source draft",
    ):
        _validate(tmp_path, data)


def test_transcript_pattern_matching_source_draft_is_accepted(tmp_path: Path) -> None:
    data = _valid_transcript()
    data["event"]["assistance_emergence_pattern"] = _source_draft_pattern(data)
    _validate(tmp_path, data)


def test_exchange_holding_only_a_tool_call_is_rejected(tmp_path: Path) -> None:
    data = copy.deepcopy(
        yaml.safe_load(FIXTURE_TRANSCRIPT_PATH.read_text(encoding="utf-8"))
    )
    load_forced_event_transcript(_write_transcript(tmp_path, data))

    index, exchange = next(
        (index, exchange)
        for index, exchange in enumerate(data["exchanges"])
        if len(exchange["items"]) > 1
        and exchange["items"][0]["type"] == "pa_tool_call"
    )
    # Same items, same order: the PA's call is lifted into an exchange of its
    # own, so that exchange carries no PA-visible text.
    data["exchanges"][index : index + 1] = [
        {"items": exchange["items"][:1]},
        {"items": exchange["items"][1:]},
    ]

    with pytest.raises(
        ForcedEventTranscriptLoadError,
        match="every exchange must carry a PA-visible spoken line",
    ):
        load_forced_event_transcript(_write_transcript(tmp_path, data))


def test_exchange_holding_only_an_environment_action_is_rejected(
    tmp_path: Path,
) -> None:
    """A delivered third-party message does not carry an exchange on its own.

    Editorial (owner ruling 2026-09-13, O2): an exchange is a unit of dialogue,
    and a delivery is not dialogue. Somebody speaks after something arrives --
    the user or the PA -- and an exchange that holds only the arrival is a
    transcript defect, whatever it later contributes to memory (and under the
    pool it contributes an `[environment_event]` or `[third_party]` block).
    """
    data = _fixture_transcript()
    load_forced_event_transcript(_write_transcript(tmp_path, data))

    index, exchange = next(
        (index, exchange)
        for index, exchange in enumerate(data["exchanges"])
        if len(exchange["items"]) > 1
        and exchange["items"][0]["type"] == "environment_action"
    )
    assert exchange["items"][0]["visibility"] in {"pa", "both"}
    # Same items, same order: the delivery is lifted into an exchange of its
    # own, so that exchange carries no PA-visible message.
    data["exchanges"][index : index + 1] = [
        {"items": exchange["items"][:1]},
        {"items": exchange["items"][1:]},
    ]

    with pytest.raises(
        ForcedEventTranscriptLoadError,
        match="every exchange must carry a PA-visible spoken line",
    ):
        load_forced_event_transcript(_write_transcript(tmp_path, data))


def test_exchange_holding_only_a_third_party_message_is_rejected(
    tmp_path: Path,
) -> None:
    """A third party speaking as a `message` item carries no exchange either.

    A `MessageItem` whose `source` is a `SourceRef` rather than the string
    `user` or `pa` is neither the person's speech nor the PA's reply, so it is
    not the spoken line the editorial rule asks for -- the same call the rule
    makes on a delivered `environment_action`. The rule therefore has to read
    the source, not just the item type.
    """
    data = _fixture_transcript()
    action = _items_of_type(data, "environment_action")[0]
    index, exchange = next(
        (index, exchange)
        for index, exchange in enumerate(data["exchanges"])
        if action in exchange["items"]
    )
    # The delivery becomes a direct third-party message -- the other way a
    # third party speaks -- and is lifted into an exchange of its own, so that
    # exchange carries no message from the user or the PA.
    exchange["items"][exchange["items"].index(action)] = {
        "type": "message",
        "offset_seconds": action["offset_seconds"],
        "source": {"kind": "third_party", "id": "Glenmont Trail Office"},
        "visibility": "both",
        "content": action["content"],
    }
    data["exchanges"][index : index + 1] = [
        {"items": exchange["items"][:1]},
        {"items": exchange["items"][1:]},
    ]
    data["assertions"]["expected_environment_actions"] = []

    with pytest.raises(
        ForcedEventTranscriptLoadError,
        match="every exchange must carry a PA-visible spoken line",
    ):
        load_forced_event_transcript(_write_transcript(tmp_path, data))


def test_authored_held_attempt_is_rejected(tmp_path: Path) -> None:
    """A forced transcript never authors the attempt the gate stops.

    The gate opens strictly before the behaviour the event is about, so a
    pre-reveal attempt would be a second thing going wrong in a story that has
    room for one. The rule sits in the authoring layer that both
    `generate_forced_event_transcript.py` and
    `scripts/validate_forced_event_transcript.py` come through, so it covers a
    generated transcript and a hand-written one alike;
    `load_forced_event_transcript` is left alone because the executor and
    `forced_event_replay` keep the held-call support they share with the
    accumulation line.
    """
    # `evolv_01` is the gated one, and its first call sits before the reveal,
    # so the hold below is the shape the old prompt asked for rather than one
    # an earlier ordering rule would reject on its way past.
    gated_path = (
        PROJECT_ROOT
        / "Authoring/sessions/forced/transcripts/user_a/evolv_01/transcript.yaml"
    )
    data = copy.deepcopy(yaml.safe_load(gated_path.read_text(encoding="utf-8")))
    _validate(tmp_path, data)

    # The shape the old prompt asked for: the gated write composed once before
    # the reveal and again after it. Every other rule accepts it -- the attempt
    # is on the gate tool, it is before the reveal, it has its later executed
    # call, and the executed count is untouched -- so this rule is the only
    # thing that stands between the corpus and two things going wrong.
    held = copy.deepcopy(data)
    gated = next(
        item
        for item in _items_of_type(held, "pa_tool_call")
        if item["name"] == "mcp_state_email_draft_create"
    )
    attempt = copy.deepcopy(gated)
    attempt["call_id"] = f"{gated['call_id']}_held"
    attempt["offset_seconds"] = 10
    attempt["held"] = True
    held["exchanges"][0]["items"].append(attempt)
    held["assertions"]["held_tool_calls"] = [attempt["call_id"]]
    with pytest.raises(
        ForcedEventTranscriptLoadError, match="forced_transcript_held_attempt"
    ):
        _validate(tmp_path, held)

    # The two fields are locked to each other in the model validator, so a
    # declared hold with no held item never reaches the rule above -- it is
    # already an error one layer earlier, under its own message.
    declared = copy.deepcopy(data)
    declared["assertions"]["held_tool_calls"] = [
        _items_of_type(declared, "pa_tool_call")[0]["call_id"]
    ]
    with pytest.raises(
        ForcedEventTranscriptLoadError,
        match="items marked held must exactly match",
    ):
        _validate(tmp_path, declared)


def test_exchange_requires_a_spoken_line_and_claims_nothing_about_memory(
    tmp_path: Path,
) -> None:
    """The exchange rule survives the memory-pool ruling as an editorial rule.

    Under the pool a tool call and a third party's words both reach memory, so
    the rule can no longer be justified by what projects to nothing, and its
    error text must make no claim about memory at all. What it still holds is
    editorial: an exchange is a unit of dialogue and must contain some.
    """
    data = _fixture_transcript()
    index, exchange = next(
        (index, exchange)
        for index, exchange in enumerate(data["exchanges"])
        if len(exchange["items"]) > 1
        and exchange["items"][0]["type"] == "pa_tool_call"
    )
    data["exchanges"][index : index + 1] = [
        {"items": exchange["items"][:1]},
        {"items": exchange["items"][1:]},
    ]

    with pytest.raises(ForcedEventTranscriptLoadError) as raised:
        load_forced_event_transcript(_write_transcript(tmp_path, data))
    message = str(raised.value)
    assert "every exchange must carry a PA-visible spoken line" in message
    assert "memory" not in message.casefold()
