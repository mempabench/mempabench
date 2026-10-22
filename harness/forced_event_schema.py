"""Schema and source validation for authored forced-event transcripts."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from harness.tool_refs import refs_in, rewrite_ref_producers
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from simulator.schemas import PersonaIdentity, PreferenceMatrix, YamlModelLoadError
from simulator.skills import load_actor_skill_spec
from scripts.outline_schema import CompactOutline
from scripts.session_authoring_schema import ForcedEventDraft


#: Who a forced-event item reaches. `god_only` is world setup that reaches
#: neither party, so it is the one value the text judge face drops
#: (`harness/judge_pack.py` `TEXT_JUDGE_VISIBILITY`, kept in step by a test);
#: the IX face reads the transcript unfiltered, as it does everywhere else.
Visibility = Literal["pa", "user", "both", "god_only"]


class ForcedEventTranscriptLoadError(ValueError):
    """Raised when a forced-event transcript or one of its sources is invalid."""


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EventMetadata(StrictModel):
    event_id: str = Field(min_length=1)
    persona_id: str = Field(min_length=1)
    # Marks the artifact as a world-toolbox transcript: the runtime stages the
    # full world and exposes the whole catalog (see harness/state_runtime.py).
    tool_status: Literal["world_toolbox"]
    source_event_path: str = Field(min_length=1)
    source_acc_num: int = Field(gt=0)
    context: Literal["work", "personal"]
    # The cell whose setting this event moves, spelled the way every other
    # artifact spells it. `harness/gate.py` resolves a session's gate off
    # `meta.target_cell` and nothing else (`_target_attribute`), and a forced
    # transcript is the artifact the runtime freezes and reads -- so it carries
    # the field an ACC artifact carries
    # (`scripts/session_authoring_schema.py:ForcedEventMeta.target_cell`)
    # rather than leaving the runtime to derive one. Without it the gateless
    # branch raised on every forced event whose attribute is compressed, which
    # is three of the five.
    target_cell: str = Field(pattern=r"^(work|personal)_[a-z0-9_]+$")
    target_attribute: str = Field(min_length=1)
    pre_setting: str = Field(min_length=1)
    post_setting: str = Field(min_length=1)
    # The opening pattern the source draft realizes (its
    # simulation.assistance_emergence.pattern); the generator writes it and
    # the validator checks it against the draft. Optional only for transcripts
    # written before this field existed.
    assistance_emergence_pattern: str | None = None

    @model_validator(mode="after")
    def _target_cell_matches_context_and_attribute(self) -> EventMetadata:
        """The three identity fields say one thing, so they have to agree.

        Same check `ForcedEventMeta._validate_forced_identity` makes on the ACC
        side, extended to the attribute because this block spells it out instead
        of deriving it. `_validate_repo_sources` then binds the derived cell to
        the source draft and to the outline entry, so a declared `target_cell`
        cannot drift from either.
        """
        expected = f"{self.context}_{self.target_attribute}"
        if self.target_cell != expected:
            raise ValueError(
                "target_cell must match context and target_attribute: "
                f"target_cell={self.target_cell!r} expected={expected!r}"
            )
        return self

    @field_validator("source_event_path")
    @classmethod
    def _source_path_stays_relative(cls, value: str) -> str:
        path = Path(value)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(
                "source_event_path must be a repo-relative path without '..'"
            )
        return value


class FixturePrecondition(StrictModel):
    path: str = Field(min_length=1)
    expectation: Literal["present"]

    @field_validator("path")
    @classmethod
    def _fixture_path_stays_relative(cls, value: str) -> str:
        path = Path(value)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("fixture path must be relative and must not contain '..'")
        return value


class Preconditions(StrictModel):
    required_fixtures: list[FixturePrecondition] = Field(min_length=1)


class SourceRef(StrictModel):
    kind: Literal["third_party", "environment"]
    id: str = Field(min_length=1)


def _content_attributes_source(content: str, source_id: str) -> bool:
    source_tokens = [
        token
        for token in re.split(r"[^a-z0-9]+", source_id.casefold())
        if len(token) >= 3
    ]
    lowered = content.casefold()
    return bool(source_tokens) and any(token in lowered for token in source_tokens)


class MessageItem(StrictModel):
    type: Literal["message"]
    offset_seconds: int = Field(ge=0)
    source: Literal["user", "pa"] | SourceRef
    visibility: Visibility
    content: str = Field(min_length=1)
    # The `task_facts` keys this user message dictates. It is the transcript's
    # half of `reveal_at`: the draft says which *beat* reveals a fact, and a
    # transcript has no beats, so it says which *message* does. The executor
    # opens the gate marker at this message and nowhere else -- an earlier user
    # message that happens to carry the same terms is not the reveal.
    #
    # A message id would be the other way to write this; naming the fact on the
    # message itself survives the generator's offset re-sequencing, which a
    # positional index would not.
    reveals: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_source_visibility(self) -> MessageItem:
        if self.reveals and self.source != "user":
            raise ValueError("only a user message can reveal a task_fact")
        if len(self.reveals) != len(set(self.reveals)):
            raise ValueError("duplicate reveals entries are not allowed")
        if (
            self.visibility == "god_only"
            and isinstance(self.source, str)
            and self.source in {"user", "pa"}
        ):
            raise ValueError(
                "a user or pa message cannot use god_only visibility"
            )
        if (
            isinstance(self.source, SourceRef)
            and self.source.kind == "third_party"
            and not _content_attributes_source(self.content, self.source.id)
        ):
            raise ValueError(
                "third-party message content must include visible source attribution"
            )
        return self


class PAToolCallItem(StrictModel):
    type: Literal["pa_tool_call"]
    offset_seconds: int = Field(ge=0)
    call_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    arguments: dict[str, Any]
    # The world answers live at replay time (record ids and counters depend on
    # run history, so exact results cannot be authored). For causally
    # load-bearing calls the writer may pin a stable subset the live result
    # must contain (e.g. a forecast recommendation).
    expected_contains: Any | None = None
    # A call the session's gate stops before it runs. The accumulation line
    # produces this shape live -- the PA composes the gated write before the
    # user has dictated it, the store refuses it, and the call reaches the
    # record as held rather than executed. A forced transcript is authored, so
    # the writer says which call that is; the executor still checks the claim
    # against the live gate rather than taking it (`ForcedEventExecutor`).
    held: bool = False


class EnvironmentActionItem(StrictModel):
    type: Literal["environment_action"]
    offset_seconds: int = Field(ge=0)
    action_id: str = Field(min_length=1)
    action: Literal["deliver_message"]
    source: SourceRef
    visibility: Visibility
    content: str = Field(min_length=1)
    # Third-party messages land in the world as well as in the conversation:
    # ``email`` writes an inbox record, ``message`` an SMS thread record.
    channel: Literal["email", "message"] | None = None
    sender: str | None = None
    subject: str | None = None

    @model_validator(mode="after")
    def _validate_attribution(self) -> EnvironmentActionItem:
        if (
            self.source.kind == "third_party"
            and not _content_attributes_source(self.content, self.source.id)
        ):
            raise ValueError(
                "third-party message content must include visible source attribution"
            )
        if self.source.kind == "third_party" and self.channel is None:
            raise ValueError(
                "third-party environment actions must declare channel (email|message)"
            )
        return self


TranscriptItem = Annotated[
    MessageItem | PAToolCallItem | EnvironmentActionItem,
    Field(discriminator="type"),
]


class Exchange(StrictModel):
    items: list[TranscriptItem] = Field(min_length=1)

    @model_validator(mode="after")
    def _require_pa_visible_text(self) -> Exchange:
        # An exchange is a unit of dialogue and must contain some: a spoken
        # line from the user or the PA, which a `message` item whose `source`
        # is the string `user` or `pa` is. This is an editorial rule about how
        # a transcript is written, and nothing else -- it makes no claim about
        # what does or does not reach memory (owner ruling 2026-09-13, O2).
        if not any(
            isinstance(item, MessageItem)
            and isinstance(item.source, str)
            and item.visibility in {"pa", "both"}
            and item.content.strip()
            for item in self.items
        ):
            raise ValueError(
                "every exchange must carry a PA-visible spoken line from the "
                "user or the PA: neither a tool call nor a third-party "
                "message, delivered or spoken, stands alone in an exchange"
            )
        return self


class PersistentStateDiffExpectation(StrictModel):
    changed_paths: list[str] = Field(min_length=1)
    state_diff_count: int = Field(ge=1)

    @field_validator("changed_paths")
    @classmethod
    def _paths_stay_relative(cls, values: list[str]) -> list[str]:
        if len(values) != len(set(values)):
            raise ValueError("changed_paths must be unique")
        for value in values:
            path = Path(value)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(
                    "changed_paths entries must be relative and must not contain '..'"
                )
        return values


class Assertions(StrictModel):
    required_tool_calls: list[str] = Field(default_factory=list)
    # The call_ids the gate is expected to stop. They are declared in the
    # transcript and never executed, so they are the one set of call ids that
    # stays out of `required_tool_calls`.
    held_tool_calls: list[str] = Field(default_factory=list)
    forbidden_tool_names: list[str] = Field(default_factory=list)
    expected_persistent_state_diff: Literal["none"] | PersistentStateDiffExpectation
    expected_environment_actions: list[str] = Field(default_factory=list)


class ForcedEventTranscript(StrictModel):
    event: EventMetadata
    preconditions: Preconditions
    # No count floor. The old `min_length=7` said a forced event is at least
    # seven exchanges, which is a length rule standing in for a story rule; the
    # real requirement is causal completeness (the transcript covers every
    # draft beat, the wrong behaviour has its consequence, and the user states
    # the rule last), checked against the draft in `_validate_repo_sources`.
    exchanges: list[Exchange] = Field(min_length=1)
    # The gated values this event's user dictates, copied verbatim from the
    # source draft's `simulation.task_facts` / `simulation.reveal_at`. The
    # transcript is the artifact the runtime freezes, so `harness/gate.py`
    # resolves its gate off these -- `_gate_source` already reads a forced
    # transcript's copy at the top level.
    task_facts: dict[str, str] = Field(default_factory=dict)
    reveal_at: dict[str, str] = Field(default_factory=dict)
    assertions: Assertions
    # Deterministic first-person memory record of this event, compiled at
    # authoring time from the draft's causality and appended verbatim to the
    # user simulator's history after replay. It is the character's own diary
    # line, not an authoring note: no attribute or setting names, no
    # evaluation vocabulary, no instruction to raise the event again later.
    user_memory_seed: str = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_user_memory_seed(self) -> ForcedEventTranscript:
        seed = self.user_memory_seed.strip()
        sentences = [part for part in re.split(r"[.!?]+", seed) if part.strip()]
        if not 2 <= len(sentences) <= 4:
            raise ValueError(
                "user_memory_seed must be 2-4 sentences, "
                f"got {len(sentences)}"
            )
        if not re.search(r"\b(?:I|I'\w+|me|my|mine|myself)\b", seed):
            raise ValueError("user_memory_seed must be written in the first person")
        if _contains_authoring_metadata(seed, self.event) or _contains_seed_metadata(
            seed, self.event
        ):
            raise ValueError(
                "user_memory_seed must not name the attribute or settings, use "
                "evaluation vocabulary, or promise to raise the event again"
            )
        return self

    @model_validator(mode="after")
    def _validate_event_sequence(self) -> ForcedEventTranscript:
        items = [item for exchange in self.exchanges for item in exchange.items]
        offsets = [item.offset_seconds for item in items]
        if any(current <= previous for previous, current in zip(offsets, offsets[1:])):
            raise ValueError(
                "item offset_seconds must be strictly increasing across the whole event"
            )

        tool_calls = [item for item in items if isinstance(item, PAToolCallItem)]
        call_ids = [item.call_id for item in tool_calls]
        _require_unique(call_ids, "call_id")
        _require_unique(self.assertions.required_tool_calls, "required_tool_calls")
        _require_unique(self.assertions.held_tool_calls, "held_tool_calls")
        held_ids = {item.call_id for item in tool_calls if item.held}
        if held_ids != set(self.assertions.held_tool_calls):
            raise ValueError(
                "pa_tool_call items marked held must exactly match "
                "assertions.held_tool_calls"
            )
        if set(call_ids) - held_ids != set(self.assertions.required_tool_calls):
            raise ValueError(
                "pa_tool_call call_id values must exactly match "
                "assertions.required_tool_calls, held calls excepted"
            )

        action_items = [
            item for item in items if isinstance(item, EnvironmentActionItem)
        ]
        action_ids = [item.action_id for item in action_items]
        _require_unique(action_ids, "action_id")
        _require_unique(
            self.assertions.expected_environment_actions,
            "expected_environment_actions",
        )
        if set(action_ids) != set(self.assertions.expected_environment_actions):
            raise ValueError(
                "environment action_id values must exactly match "
                "assertions.expected_environment_actions"
            )

        # PA-visibility, the two places it decides the event rather than just
        # dressing it. A `visibility: user` message never enters the PA
        # session, so a reveal written that way would open the gate on
        # information the PA was never given -- the write would be released by
        # words it cannot have read. And a forced event ends on the user
        # stating the new rule: said out of the PA's hearing, the rule is not
        # in the session the run is about, and every later probe measures a PA
        # that was never told. Both are errors here, and the executor refuses
        # the first again at replay
        # (`harness.forced_event_replay.GatedReplay.observe_message`).
        messages = [item for item in items if isinstance(item, MessageItem)]
        for item in messages:
            if item.reveals and item.visibility not in {"pa", "both"}:
                raise ValueError(
                    "a message that reveals a task_fact must be PA-visible: "
                    f"reveals={item.reveals!r} visibility={item.visibility!r}"
                )
        if not messages or messages[-1].source != "user" or messages[
            -1
        ].visibility not in {"pa", "both"}:
            raise ValueError(
                "a forced event ends on the user stating the new rule, so the "
                "last message must come from the user and be PA-visible"
            )

        revealed: dict[str, int] = {}
        for position, item in enumerate(items):
            if not isinstance(item, MessageItem):
                continue
            for key in item.reveals:
                if key not in self.reveal_at:
                    raise ValueError(
                        f"message reveals task_fact {key!r}, which reveal_at "
                        "does not place"
                    )
                if key in revealed:
                    raise ValueError(
                        f"task_fact {key!r} is revealed by more than one message"
                    )
                revealed[key] = position
        for key in self.reveal_at:
            if key not in self.task_facts:
                raise ValueError(f"reveal_at.{key} names no key in task_facts")
            if key not in revealed:
                raise ValueError(
                    f"reveal_at places task_fact {key!r}, but no user message "
                    "in the transcript reveals it"
                )
        # A held call is a call the gate stopped, so it belongs before the
        # message that opens the gate. After the reveal nothing holds it.
        first_reveal = min(revealed.values(), default=None)
        if first_reveal is not None:
            for position, item in enumerate(items):
                if (
                    isinstance(item, PAToolCallItem)
                    and item.held
                    and position > first_reveal
                ):
                    raise ValueError(
                        f"held call {item.call_id!r} comes after the reveal, "
                        "so nothing holds it"
                    )

        forbidden = set(self.assertions.forbidden_tool_names)
        used_forbidden = sorted(forbidden.intersection(item.name for item in tool_calls))
        if used_forbidden:
            raise ValueError(
                f"transcript uses forbidden tool names: {used_forbidden}"
            )

        for item in items:
            if (
                isinstance(item, (MessageItem, EnvironmentActionItem))
                and item.visibility in {"pa", "both"}
                and _contains_authoring_metadata(item.content, self.event)
            ):
                raise ValueError(
                    "PA-visible content must not expose authoring metadata or the "
                    "preference shift"
                )
        return self


def _require_unique(values: list[str], field_name: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"duplicate {field_name} values are not allowed")


def _contains_authoring_metadata(content: str, event: EventMetadata) -> bool:
    escaped_pre = re.escape(event.pre_setting)
    escaped_post = re.escape(event.post_setting)
    patterns = (
        rf"\b{escaped_pre}\s*(?:->|→)\s*{escaped_post}\b",
        r"\bauthoring[ _-]*(?:rationale|instruction|script|note)\b",
        r"\bgod[ _-]*(?:only|intervention|goal|action)\b",
        r"\b(?:target_attribute|pre_setting|post_setting)\b",
    )
    return any(re.search(pattern, content, flags=re.IGNORECASE) for pattern in patterns)


# Vocabulary that would betray the memory seed as authored for a benchmark, or
# that would push the simulator to volunteer the event in a later session.
_SEED_FORBIDDEN_PATTERNS = (
    r"\bpreferences?\b",
    r"\bsettings?\b",
    r"\bassistant\s+behaviou?r\b",
    r"\bprobes?\b",
    r"\bevaluation\b",
    r"\bbenchmark\b",
    r"\bremind\w*\b",
    r"\bbring\s+(?:this|it|that)\s+up\b",
    r"\bI(?:'ll|\s+will|\s+am\s+going\s+to|\s+plan\s+to)\s+\w*\s*"
    r"(?:mention|raise|repeat|bring)\b",
)


def _contains_seed_metadata(content: str, event: EventMetadata) -> bool:
    insensitive = (
        *_SEED_FORBIDDEN_PATTERNS,
        rf"\b{re.escape(event.target_attribute)}\b",
        rf"\b{re.escape(event.target_attribute.replace('_', ' '))}\b",
    )
    # Setting labels match case-sensitively, as in the forced-event draft
    # validator: the label ``Reactive`` is authoring vocabulary, while the
    # ordinary adjective "reactive" is how a person would actually write it.
    sensitive = (
        rf"\b{re.escape(event.pre_setting)}\b",
        rf"\b{re.escape(event.post_setting)}\b",
    )
    return any(
        re.search(pattern, content, flags=re.IGNORECASE) for pattern in insensitive
    ) or any(re.search(pattern, content) for pattern in sensitive)


def load_forced_event_transcript(path: Path | str) -> ForcedEventTranscript:
    source_path = Path(path)
    data = _load_yaml_mapping(source_path, label="forced-event transcript")
    try:
        return ForcedEventTranscript.model_validate(data)
    except ValidationError as exc:
        raise ForcedEventTranscriptLoadError(
            f"Invalid forced-event transcript: path={source_path} error={exc}"
        ) from exc


def validate_forced_event_transcript(
    path: Path | str,
    *,
    project_root: Path | str = ".",
) -> ForcedEventTranscript:
    transcript = load_forced_event_transcript(path)
    root = Path(project_root).resolve()
    try:
        _validate_repo_sources(transcript, project_root=root)
    except ForcedEventTranscriptLoadError:
        raise
    except (KeyError, TypeError, ValueError, YamlModelLoadError) as exc:
        raise ForcedEventTranscriptLoadError(
            f"Forced-event source validation failed: path={path} error={exc}"
        ) from exc
    return transcript


def _validate_repo_sources(
    transcript: ForcedEventTranscript,
    *,
    project_root: Path,
) -> None:
    event = transcript.event
    expected_source_path = Path(
        f"Authoring/sessions/forced/drafts/{event.persona_id}/{event.event_id}.yaml"
    )
    if Path(event.source_event_path) != expected_source_path:
        raise ForcedEventTranscriptLoadError(
            "source_event_path does not match persona_id/event_id: "
            f"expected={expected_source_path} actual={event.source_event_path}"
        )

    event_draft_path = _resolve_repo_path(
        project_root,
        event.source_event_path,
        field_name="source_event_path",
    )
    draft_data = _load_yaml_mapping(event_draft_path, label="forced event draft")
    try:
        draft = ForcedEventDraft.model_validate(draft_data)
    except ValidationError as exc:
        raise ForcedEventTranscriptLoadError(
            f"source forced event draft is invalid: path={event_draft_path} error={exc}"
        ) from exc
    meta = draft.event_meta
    if meta.tool_status != "world_toolbox":
        raise ForcedEventTranscriptLoadError(
            "source forced event draft must have tool_status=world_toolbox"
        )
    draft_pattern = draft.simulation.assistance_emergence.pattern
    if (
        event.assistance_emergence_pattern is not None
        and event.assistance_emergence_pattern != draft_pattern
    ):
        raise ForcedEventTranscriptLoadError(
            "event.assistance_emergence_pattern does not match the source draft: "
            f"transcript={event.assistance_emergence_pattern!r} draft={draft_pattern!r}"
        )

    for field_name, setting in (
        ("pre_setting", event.pre_setting),
        ("post_setting", event.post_setting),
    ):
        try:
            load_actor_skill_spec(
                event.target_attribute,
                setting,
                root=project_root / "data/actor_skill_specs",
            )
        except FileNotFoundError as exc:
            raise ForcedEventTranscriptLoadError(
                f"{field_name} has no actor skill spec: "
                f"attribute={event.target_attribute!r} setting={setting!r}"
            ) from exc

    expected_meta = {
        "event_id": event.event_id,
        "persona_id": event.persona_id,
        "source_acc_num": event.source_acc_num,
        "target_cell": f"{event.context}_{event.target_attribute}",
        "pre_setting": event.pre_setting,
        "post_setting": event.post_setting,
        "context": event.context,
    }
    for field, expected in expected_meta.items():
        actual = getattr(meta, field)
        if actual != expected:
            transcript_field = field
            raise ForcedEventTranscriptLoadError(
                f"{transcript_field} does not match event draft: "
                f"expected={actual!r} actual={expected!r}"
            )

    expected_identity = Path(f"data/personas/{event.persona_id}_identity.yaml")
    identity_path = _resolve_repo_path(
        project_root, str(expected_identity), field_name="identity_path"
    )
    identity = PersonaIdentity.from_yaml(identity_path)

    preference_relative = Path(
        f"data/personas/{event.persona_id}_preferences.yaml"
    )
    preference_path = _resolve_repo_path(
        project_root, str(preference_relative), field_name="preference_path"
    )
    matrix = PreferenceMatrix.from_yaml(preference_path)
    if matrix.character != identity.name:
        raise ForcedEventTranscriptLoadError(
            "identity and preference matrix describe different personas: "
            f"identity={identity.name!r} matrix={matrix.character!r}"
        )
    try:
        matrix_setting = matrix.get(event.context, event.target_attribute).setting
    except KeyError as exc:
        raise ForcedEventTranscriptLoadError(
            "target_attribute is missing from the persona preference matrix: "
            f"context={event.context!r} attribute={event.target_attribute!r}"
        ) from exc
    if matrix_setting != event.pre_setting:
        raise ForcedEventTranscriptLoadError(
            "pre_setting does not match the persona preference matrix: "
            f"expected={matrix_setting!r} actual={event.pre_setting!r}"
        )

    outline_relative = Path(
        f"Authoring/outline/drafts/{event.persona_id}/outline.yaml"
    )
    outline_path = _resolve_repo_path(
        project_root, str(outline_relative), field_name="outline_path"
    )
    outline_data = _load_yaml_mapping(outline_path, label="session outline")
    try:
        outline = CompactOutline.model_validate(outline_data)
    except ValidationError as exc:
        raise ForcedEventTranscriptLoadError(
            f"accepted outline is invalid: path={outline_path} error={exc}"
        ) from exc
    if outline.meta.persona != event.persona_id:
        raise ForcedEventTranscriptLoadError(
            "outline persona does not match transcript persona_id: "
            f"expected={event.persona_id!r} actual={outline.meta.persona!r}"
        )
    matching_entries = [
        entry
        for entry in outline.sessions
        if entry.acc_num == event.source_acc_num
    ]
    if len(matching_entries) != 1:
        raise ForcedEventTranscriptLoadError(
            "outline must contain exactly one source_acc_num entry: "
            f"source_acc_num={event.source_acc_num} matches={len(matching_entries)}"
        )
    outline_entry = matching_entries[0]
    observed_outline = {
        "session_type": outline_entry.identity.session_type,
        "context": outline_entry.context,
        "target_cell": outline_entry.identity.target_cell,
    }
    expected_outline = {
        "session_type": "evolving_event",
        "context": event.context,
        "target_cell": f"{event.context}_{event.target_attribute}",
    }
    for field, expected in expected_outline.items():
        actual = observed_outline[field]
        if actual != expected:
            raise ForcedEventTranscriptLoadError(
                f"outline {field} does not match transcript/event draft: "
                f"expected={expected!r} actual={actual!r}"
            )

    arc = outline.evolving_map.get(outline_entry.identity.timeline_slot)
    if arc is None:
        raise ForcedEventTranscriptLoadError(
            "outline evolving_map is missing the event timeline_slot: "
            f"timeline_slot={outline_entry.identity.timeline_slot!r}"
        )
    expected_evolving = {
        "target_cell": f"{event.context}_{event.target_attribute}",
        "pre_setting": event.pre_setting,
        "post_setting": event.post_setting,
    }
    for field, expected in expected_evolving.items():
        actual = getattr(arc, field)
        if actual != expected:
            raise ForcedEventTranscriptLoadError(
                f"outline evolving_map {field} does not match transcript: "
                f"expected={expected!r} actual={actual!r}"
            )

    # World-toolbox model: preconditions name the world files the event relies
    # on, relative to the persona fixture root; each must actually exist.
    fixture_prefix = Path("data/state_fixtures") / event.persona_id
    for item in transcript.preconditions.required_fixtures:
        _resolve_repo_path(
            project_root,
            (fixture_prefix / item.path).as_posix(),
            field_name=f"preconditions.required_fixtures[{item.path}]",
        )

    expected_diff = transcript.assertions.expected_persistent_state_diff
    state_changes = draft.task_execution.state_changes
    if not state_changes and expected_diff != "none":
        raise ForcedEventTranscriptLoadError(
            "expected_persistent_state_diff must be none when the event draft declares "
            "no state changes"
        )
    if state_changes and expected_diff == "none":
        raise ForcedEventTranscriptLoadError(
            "expected_persistent_state_diff must declare changed paths when the "
            "event draft declares state changes"
        )
    if expected_diff != "none":
        if expected_diff.state_diff_count != len(state_changes):
            raise ForcedEventTranscriptLoadError(
                "expected_persistent_state_diff.state_diff_count must match the "
                "event draft state change count"
            )

    validate_gated_values_match_draft(transcript, draft)

    # Causal completeness, in place of the retired seven-exchange floor: the
    # transcript has to have room for every beat the draft authored, and the
    # user's stated rule is what ends the event.
    draft_beats = draft.simulation.beats
    if len(transcript.exchanges) < len(draft_beats):
        raise ForcedEventTranscriptLoadError(
            "transcript does not cover every draft beat: "
            f"beats={len(draft_beats)} exchanges={len(transcript.exchanges)}"
        )
    all_items = [item for exchange in transcript.exchanges for item in exchange.items]
    visible = [
        item
        for item in all_items
        if isinstance(item, MessageItem) and item.visibility in {"pa", "both"}
    ]
    if not visible or visible[-1].source != "user":
        raise ForcedEventTranscriptLoadError(
            "a forced event ends on the user stating the new rule, so the last "
            "PA-visible message must come from the user"
        )

    expected_calls = [
        step for step in draft.task_execution.atomic_steps if step.tool
    ]
    all_calls = [item for item in all_items if isinstance(item, PAToolCallItem)]
    held_calls = [item for item in all_calls if item.held]
    actual_calls = [item for item in all_calls if not item.held]
    # `forced_transcript_held_attempt`: a forced transcript never authors the
    # attempt the gate stops. The gate opens strictly before the behaviour the
    # event exists to show ("One thing goes wrong in this story, and it is the
    # behavior", `data/scripts/prompts/forced_event_gen_system.md`), so a
    # pre-reveal attempt would be a second thing going wrong. This sits in the
    # authoring layer, not in the model validator: `generate_forced_event_
    # transcript.py` and `scripts/validate_forced_event_transcript.py` both
    # come through here, so it covers a generated transcript and a hand-written
    # one alike, while `load_forced_event_transcript` stays as it was and the
    # runtime keeps the held-call support it shares with the accumulation line.
    if held_calls or transcript.assertions.held_tool_calls:
        raise ForcedEventTranscriptLoadError(
            "forced_transcript_held_attempt: a forced transcript never authors "
            "a held attempt; the gated step is written once, after the reveal, "
            "carrying the user's wording: "
            f"held={[item.call_id for item in held_calls]!r} "
            f"held_tool_calls={transcript.assertions.held_tool_calls!r}"
        )
    # The binding, explicitly: one draft atomic step is one executed call, the
    # gated step included -- it is written once, after the reveal, so nothing
    # authored stands beside it. `held_calls` is empty by the rule above;
    # `_validate_held_calls` below is what the runtime's own held records are
    # still checked against.
    if len(actual_calls) != len(expected_calls):
        raise ForcedEventTranscriptLoadError(
            "transcript executed tool call count does not match the forced "
            f"event's tool steps: expected={len(expected_calls)} "
            f"actual={len(actual_calls)}"
        )
    _validate_held_calls(
        held_calls,
        actual_calls,
        draft_path=event_draft_path,
        project_root=project_root,
    )
    authored_argument_fields_by_tool = {
        "document_create": {"content"},
        "document_update": {"replacement"},
        "email_draft_create": {"subject", "body"},
        "email_message_send": {"subject", "body"},
        "message_send": {"body"},
    }
    step_to_call = {
        expected.step_id: actual.call_id
        for actual, expected in zip(actual_calls, expected_calls, strict=True)
    }
    for item in draft.intervention_plan:
        if item.kind == "third_party_message":
            for step_id in item.atomic_step_ids or []:
                step_to_call.setdefault(step_id, item.intervention_id)
    for index, (actual, expected) in enumerate(
        zip(actual_calls, expected_calls, strict=True), start=1
    ):
        expected_runtime_name = f"mcp_state_{expected.tool}"
        planned_arguments = rewrite_ref_producers(
            dict(expected.tool_arguments), step_to_call
        )
        if actual.name != expected_runtime_name:
            raise ForcedEventTranscriptLoadError(
                f"tool call {index} name does not match its draft tool step: "
                f"expected={expected_runtime_name!r} actual={actual.name!r}"
            )
        authored_argument_fields = authored_argument_fields_by_tool.get(
            expected.tool, set()
        )
        if any(
            actual.arguments.get(name) != value
            for name, value in planned_arguments.items()
            if name not in authored_argument_fields
        ):
            raise ForcedEventTranscriptLoadError(
                f"tool call {index} arguments do not match the draft step's "
                "planned tool_arguments"
            )
        for name in authored_argument_fields:
            value = actual.arguments.get(name)
            if name in expected.tool_arguments and (
                not isinstance(value, str) or not value.strip()
            ):
                raise ForcedEventTranscriptLoadError(
                    f"tool call {index} transcript-authored argument {name!r} "
                    "must be non-empty text"
                )

    expected_environment_actions = [
        item.intervention_id
        for item in draft.intervention_plan
        if item.kind in {"environment_action", "third_party_message"}
    ]
    if transcript.assertions.expected_environment_actions != expected_environment_actions:
        raise ForcedEventTranscriptLoadError(
            "expected_environment_actions must match visible environment/third-party "
            "interventions in event-draft order: "
            f"expected={expected_environment_actions!r} "
            f"actual={transcript.assertions.expected_environment_actions!r}"
        )


def depended_read_call_ids(
    transcript: ForcedEventTranscript,
    draft: ForcedEventDraft,
) -> set[str]:
    """The read calls the rest of the event turns on, decided mechanically.

    Never by reading the story. A read is depended on when:

    * a later tool call names it in a `$<call_id>.<field>` argument, or
    * the draft step it is bound to is named by another step's `depends_on`,
      by a step's `readback_step_id`, or by a declared state change's
      `readback_step_id` -- the draft's own consequence chain.

    Only read-effect steps are covered: a write returns the ids the world
    assigned, which is exactly what must never be pinned.

    What such a call then has to assert is decided where the result's shape is
    known, in `scripts/generate_forced_event_transcript.py:check_world_replay`.
    """
    depended_steps: set[str] = set()
    for step in draft.task_execution.atomic_steps:
        depended_steps.update(step.depends_on)
        if step.readback_step_id:
            depended_steps.add(step.readback_step_id)
    for change in draft.task_execution.state_changes:
        if getattr(change, "readback_step_id", None):
            depended_steps.add(change.readback_step_id)

    referenced: set[str] = set()
    calls: list[PAToolCallItem] = []
    for exchange in transcript.exchanges:
        for item in exchange.items:
            if isinstance(item, PAToolCallItem):
                calls.append(item)
                for producer, _path in refs_in(dict(item.arguments)).values():
                    referenced.add(producer)

    executed = [item for item in calls if not item.held]
    read_steps = [step for step in draft.task_execution.atomic_steps if step.tool]
    depended: set[str] = set()
    for actual, expected in zip(executed, read_steps, strict=True):
        if expected.effect != "read":
            continue
        if expected.step_id in depended_steps or actual.call_id in referenced:
            depended.add(actual.call_id)
    return depended


def validate_gated_values_match_draft(
    transcript: ForcedEventTranscript,
    draft: ForcedEventDraft,
) -> None:
    """The transcript's gated values are the draft's own, verbatim.

    The transcript is what the runtime freezes and what `harness/gate.py`
    resolves, so a transcript that paraphrased `task_facts` would gate the
    replay on different words than the draft was validated against, and one
    that moved `reveal_at` would gate it at a different point in the story.
    """
    draft_task_facts = {
        key: str(value) for key, value in draft.simulation.task_facts.items()
    }
    if transcript.task_facts != draft_task_facts:
        raise ForcedEventTranscriptLoadError(
            "task_facts do not match the source draft's simulation.task_facts: "
            f"draft={draft_task_facts!r} transcript={transcript.task_facts!r}"
        )
    draft_reveal_at = dict(draft.simulation.reveal_at)
    if transcript.reveal_at != draft_reveal_at:
        raise ForcedEventTranscriptLoadError(
            "reveal_at does not match the source draft's simulation.reveal_at: "
            f"draft={draft_reveal_at!r} transcript={transcript.reveal_at!r}"
        )


def _validate_held_calls(
    held_calls: list[PAToolCallItem],
    executed_calls: list[PAToolCallItem],
    *,
    draft_path: Path,
    project_root: Path,
) -> None:
    """A held attempt belongs to a gated step, and to exactly one of them.

    The freeze check the executor cannot make: at replay time the executor only
    knows whether the marker is down, so it would accept a held attempt on a
    step the draft never gated. Here the gate is resolvable off the draft, so
    the pairing is checked against it.
    """
    if not held_calls:
        return
    from harness.gate import resolve_session_gate

    gate = resolve_session_gate(draft_path, project_root)
    if not gate:
        raise ForcedEventTranscriptLoadError(
            "transcript marks tool calls held, but the source draft's session "
            f"has no gate: held={[item.call_id for item in held_calls]!r}"
        )
    gate_name = f"mcp_state_{gate['tool']}"
    seen: set[str] = set()
    for item in held_calls:
        if item.name != gate_name:
            raise ForcedEventTranscriptLoadError(
                f"held call {item.call_id!r} is not on the session's gate tool: "
                f"gate_tool={gate_name!r} actual={item.name!r}"
            )
        retries = [call for call in executed_calls if call.name == gate_name]
        if not retries:
            raise ForcedEventTranscriptLoadError(
                f"held call {item.call_id!r} has no later executed call on the "
                "same draft step"
            )
        retry = retries[0].call_id
        if retry in seen:
            raise ForcedEventTranscriptLoadError(
                "a gated draft step may carry at most one held attempt: "
                f"step_call_id={retry!r}"
            )
        seen.add(retry)


def _resolve_repo_path(
    project_root: Path,
    value: str,
    *,
    field_name: str,
) -> Path:
    declared = Path(value)
    if declared.is_absolute() or ".." in declared.parts:
        raise ForcedEventTranscriptLoadError(
            f"{field_name} must be a repo-relative path without '..': {value}"
        )
    resolved = (project_root / declared).resolve()
    try:
        resolved.relative_to(project_root)
    except ValueError as exc:
        raise ForcedEventTranscriptLoadError(
            f"{field_name} escapes project_root: {value}"
        ) from exc
    if not resolved.is_file():
        raise ForcedEventTranscriptLoadError(
            f"{field_name} does not exist: {value}"
        )
    return resolved


def _require_mapping(data: dict[str, Any], field_name: str) -> dict[str, Any]:
    value = data.get(field_name)
    if not isinstance(value, dict):
        raise ForcedEventTranscriptLoadError(
            f"{field_name} must be a mapping in event draft"
        )
    return value


def _load_yaml_mapping(path: Path, *, label: str) -> dict[str, Any]:
    data = _load_yaml(path, label=label)
    if not isinstance(data, dict):
        raise ForcedEventTranscriptLoadError(
            f"{label} YAML root must be a mapping: path={path} "
            f"root_type={type(data).__name__}"
        )
    return data


def _load_yaml(path: Path, *, label: str) -> Any:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ForcedEventTranscriptLoadError(
            f"Failed to read {label}: path={path} error={type(exc).__name__}: {exc}"
        ) from exc
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ForcedEventTranscriptLoadError(
            f"Invalid {label} YAML: path={path} error={type(exc).__name__}: {exc}"
        ) from exc
