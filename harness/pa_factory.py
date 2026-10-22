"""Factory for benchmark PA instances."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from harness.state_runtime import world_toolbox_tools
from memory.adapter import MemoryAdapter
from memory.audit import MemoryIOAuditSink
from pa.agent.ix.collector import assert_action_type_coverage
from pa.agent.loop import AgentLoop
from pa.bus.queue import MessageBus

if TYPE_CHECKING:
    from harness.provider_avoidance import ProviderAvoidance


@dataclass(frozen=True)
class AssistantLLMProviderProfile:
    """Shared transport configuration for PA and consolidation providers."""

    provider_name: str
    api_key: str
    api_base: str | None = None
    #: The run's OpenRouter upstream avoidance, handed to every provider this
    #: profile creates (the PA of every step and probe branch, and File memory
    #: consolidation), so they count into and read one state file.
    provider_avoidance: ProviderAvoidance | None = None

    def create_provider(
        self,
        *,
        model: str,
        role: str,
        require_validated_role: bool = True,
        request_policy_catalog_path: Path | str | None = None,
    ):
        from pa.providers.registry import find_by_name
        from pa.providers.request_policy import (
            DEFAULT_MODEL_POLICY_CATALOG,
            resolve_model_request_policy,
        )

        spec = find_by_name(self.provider_name)
        if spec is None:
            raise ValueError(
                f"Unknown assistant LLM provider: value={self.provider_name!r}"
            )
        catalog_path = (
            request_policy_catalog_path or DEFAULT_MODEL_POLICY_CATALOG
        )
        policy = resolve_model_request_policy(
            self.provider_name,
            model,
            catalog_path=catalog_path,
        )
        if require_validated_role:
            policy.require_role(role)
        api_base = self.api_base or spec.default_api_base or None
        if spec.backend == "openai_compat":
            from pa.providers.openai_compat_provider import OpenAICompatProvider

            return OpenAICompatProvider(
                api_key=self.api_key,
                api_base=api_base,
                spec=spec,
                default_model=model,
                request_policy_catalog_path=catalog_path,
                provider_avoidance=self.provider_avoidance,
            )
        if spec.backend == "anthropic":
            from pa.providers.anthropic_provider import AnthropicProvider

            return AnthropicProvider(
                api_key=self.api_key,
                api_base=api_base,
                default_model=model,
            )
        raise ValueError(
            "Unsupported assistant provider backend: "
            f"provider_name={self.provider_name!r} backend={spec.backend!r}"
        )


def build_openai_compat_pa(
    *,
    workspace: Path,
    api_key: str,
    pa_model: str,
    api_base: str | None = None,
    pa_provider_name: str = "custom",
    mcp_servers: dict | None = None,
    restrict_to_workspace: bool = True,
    memory_adapter: MemoryAdapter | None = None,
    consolidation_llm_provider: Any | None = None,
    consolidation_model: str | None = None,
    require_committed_adapter_writes: bool = False,
    memory_io_recorder: MemoryIOAuditSink | None = None,
) -> AgentLoop:
    """Build a PA loop against any OpenAI-compatible endpoint."""
    from pa.providers.openai_compat_provider import OpenAICompatProvider
    from pa.providers.registry import find_by_name

    spec = find_by_name(pa_provider_name)
    if spec is None:
        raise RuntimeError(
            f"PA provider registry has no {pa_provider_name!r} entry"
        )
    if not api_base and not spec.default_api_base:
        raise ValueError("api_base is required for custom OpenAI-compatible providers")

    pa_llm_provider = OpenAICompatProvider(
        api_key=api_key,
        api_base=api_base,
        spec=spec,
        default_model=pa_model,
    )
    return AgentLoop(
        bus=MessageBus(),
        pa_llm_provider=pa_llm_provider,
        workspace=workspace,
        pa_model=pa_model,
        restrict_to_workspace=restrict_to_workspace,
        mcp_servers=mcp_servers or {},
        memory_adapter=memory_adapter,
        consolidation_llm_provider=consolidation_llm_provider,
        consolidation_model=consolidation_model,
        require_committed_adapter_writes=require_committed_adapter_writes,
        memory_io_recorder=memory_io_recorder,
    )


def build_pa_from_provider_profile(
    *,
    provider_profile: AssistantLLMProviderProfile,
    workspace: Path,
    pa_model: str,
    pa_max_tokens: int,
    mcp_servers: dict | None = None,
    restrict_to_workspace: bool = True,
    memory_adapter: MemoryAdapter | None = None,
    consolidation_llm_provider: Any | None = None,
    consolidation_model: str | None = None,
    require_committed_adapter_writes: bool = False,
    memory_io_recorder: MemoryIOAuditSink | None = None,
    skip_consolidation: bool = False,
) -> AgentLoop:
    pa_llm_provider = provider_profile.create_provider(
        model=pa_model,
        role="pa",
    )
    # The frozen plan's `pa_max_tokens` reaches the request payload here and
    # nowhere else: `AgentRunSpec.max_tokens` stays None on the PA path
    # (`pa/agent/loop.py` never sets it), so every PA request falls back to the
    # provider's `GenerationSettings` inside `chat_with_retry`.
    pa_llm_provider.generation = replace(
        pa_llm_provider.generation, max_tokens=pa_max_tokens
    )
    # Upstream routing and the raw-response log come from `provider_routing` in
    # configs/model_request_policies.yaml (e.g. tencent/hy4-preview only via
    # SiliconFlow). Request-body passthrough only; the request policy snapshot
    # is unchanged.
    from pa.providers.request_policy import resolve_provider_routing

    routing = resolve_provider_routing(provider_profile.provider_name, pa_model)
    if routing is not None and hasattr(pa_llm_provider, "extra_body"):
        pa_llm_provider.extra_body = routing.request_body()
    if (routing is not None and routing.raw_response_log_dir
            and hasattr(pa_llm_provider, "raw_response_log")):
        run_dir = Path(workspace).resolve().parent.parent
        ws_root = next((p for p in run_dir.parents if p.name == "workspace"), None)
        if ws_root is not None:
            pa_llm_provider.raw_response_log = (
                ws_root / routing.raw_response_log_dir / run_dir.name / "raw_pa_responses.jsonl"
            )
    return AgentLoop(
        bus=MessageBus(),
        pa_llm_provider=pa_llm_provider,
        workspace=workspace,
        pa_model=pa_model,
        restrict_to_workspace=restrict_to_workspace,
        mcp_servers=mcp_servers or {},
        memory_adapter=memory_adapter,
        consolidation_llm_provider=consolidation_llm_provider,
        consolidation_model=consolidation_model,
        require_committed_adapter_writes=require_committed_adapter_writes,
        memory_io_recorder=memory_io_recorder,
        skip_consolidation=skip_consolidation,
    )


def build_openrouter_pa(
    *,
    workspace: Path,
    openrouter_key: str,
    pa_model: str,
    mcp_servers: dict | None = None,
    restrict_to_workspace: bool = True,
    memory_adapter: MemoryAdapter | None = None,
    consolidation_llm_provider: Any | None = None,
    consolidation_model: str | None = None,
    require_committed_adapter_writes: bool = False,
    memory_io_recorder: MemoryIOAuditSink | None = None,
) -> AgentLoop:
    """Build a PA loop with the benchmark MCP tool surface."""
    return build_openai_compat_pa(
        workspace=workspace,
        api_key=openrouter_key,
        pa_model=pa_model,
        pa_provider_name="openrouter",
        restrict_to_workspace=restrict_to_workspace,
        mcp_servers=mcp_servers or {},
        memory_adapter=memory_adapter,
        consolidation_llm_provider=consolidation_llm_provider,
        consolidation_model=consolidation_model,
        require_committed_adapter_writes=require_committed_adapter_writes,
        memory_io_recorder=memory_io_recorder,
    )


def outreach_keys_from_ix_render(ix_render: Mapping[str, Any]) -> dict[str, Any]:
    """Translate the rendered `ProactiveDecision` into the two existing keys.

    The simulator branches on `outreach_decision` / `outreach_delivered`
    (`simulator/loop.py`) and must not learn to read `ix_render` — that block is
    the IX face. §4.4 makes the render the single source for this decision, so
    the harness restates it in the vocabulary the simulator already speaks.
    Neither the chosen setting nor the suppressed message crosses over.
    """
    decision = ix_render.get("proactive_decision")
    if not decision:
        return {}
    action = decision.get("action")
    if action not in {"send", "suppress"}:
        return {}
    return {"outreach_decision": action, "outreach_delivered": action == "send"}


def held_items_from_ix_render(ix_render: Mapping[str, Any]) -> list[dict[str, str]]:
    """The held ids the simulator may give a consent verdict on.

    Id plus the intent text, nothing else: no attribute, no setting, no gate
    reason. The sideband keys on `interaction_item_id` (§3.7 part 3), so an item
    only appears once the wrapper can address it by one — either a held call
    that carries it, or a held plan item that bound it.

    **Open episodes only, and only what this turn's ask named.** A verdict on an
    id the wrapper is not holding is not consent to anything: nothing can
    consume it, and if the id is ever reused a stale `allow` would be sitting
    there ready to release it. §3.7 part 3's criterion is narrower still — "every
    held id shown to the user" — so an id the ask did not name is not offered
    either: live, an ask about one cancellation came back with an `allow` for
    six ids. `ix_render.held` still lists every held call for the judge; this
    is only what the simulator may answer. A turn that opened no episode, or
    that held items without asking about them, offers the simulator nothing.
    """
    held = ix_render.get("held") or {}
    open_episode_ids = {
        str(status.get("episode_id") or "")
        for status in (ix_render.get("episodes") or {}).values()
        if not status.get("closed")
    }
    if not open_episode_ids:
        return []
    plan_remaining = held.get("plan_remaining") or []
    intent_by_plan_item = {
        item.get("id"): item.get("intent") or "" for item in plan_remaining
    }
    gate = ix_render.get("gate") or {}
    asked_ids = {str(gate.get("item_id") or "")} | {
        str(call.get("interaction_item_id") or "")
        for call in held.get("calls") or []
        if call.get("asked")
    }
    asked_ids.discard("")
    # A turn that asked nothing put no ids to the user, so there is nothing to
    # scope to: a batch or question hold opens no gate and writes no ask, and
    # its continuation protocol is the whole release mechanism (§3.4 part 3). Once a
    # turn does render an ask, only what that ask named is answerable.
    scoped = bool(asked_ids)

    items: list[dict[str, str]] = []
    seen: set[str] = set()

    def add(item_id: str, intent: str) -> None:
        if not item_id or item_id in seen or (scoped and item_id not in asked_ids):
            return
        seen.add(item_id)
        items.append({"interaction_item_id": item_id, "intent": intent})

    for call in held.get("calls") or []:
        if str(call.get("episode_id") or "") not in open_episode_ids:
            # Refused for this turn only (a fail-closed binding batch, a spent
            # release): recorded for the judge, but not something the user is
            # being asked about.
            continue
        add(
            str(call.get("interaction_item_id") or ""),
            intent_by_plan_item.get(call.get("plan_item_id"), ""),
        )
    for item in plan_remaining:
        for bound_id in item.get("bound_item_ids") or []:
            add(str(bound_id), item.get("intent") or "")
    return items


def has_pending_interaction_from_ix_render(ix_render: Mapping[str, Any]) -> bool:
    """Whether the PA is still waiting on this user after the turn.

    `held_items_from_ix_render()` answers a narrower question — which ids the
    simulator may give a *verdict* on — and a question-only or batch-only hold
    (Iterative `information_elicitation`, One-at-a-time `topic_management`)
    answers it with an empty list: nothing was addressable by an
    `interaction_item_id`, so nothing is consent-able. Using that emptiness as
    "the exchange is finished" walks the simulator out of a question, and on the
    last beat ends the session with the episode still open (§3.4 part 3 / §4.2).

    An episode stays open only while it is holding something — a call, a
    question, a batch, or a remaining plan item (`render.py:1145-1151` closes it
    the moment it holds none of them) — so "any open episode" is exactly "the PA
    is mid-interaction". This carries no text, no ids and no setting names: the
    simulator learns only that something is waiting, never what.
    """
    return any(
        not status.get("closed")
        for status in (ix_render.get("episodes") or {}).values()
    )


def reply_facts_from_ix_render(ix_render: Mapping[str, Any]) -> dict[str, Any]:
    """The wrapper-side half of the actor's record of the delivered reply.

    Owner ruling 2026-09-17: the simulated user is told how many progress
    lines and whether a closing line were delivered, and whether bracketed
    confidence tags appeared, so a reaction keyed on those can be chosen from
    fact rather than from the reply's self-description. Counts and flags off
    what was **delivered** — `notices` is the released list, `completion.
    released` is a statement about the delivered reply, a certainty with an
    empty `rendered` appended nothing. Nothing held crosses over: no held
    question, batch, notice or closing text, no hold flag, no gate, no setting.
    The executed writes and the call count are the simulator's own reading of
    `tool_events`; this key carries only what it cannot see there.
    """
    completion = ix_render.get("completion") or {}
    return {
        "notices_released": len(ix_render.get("notices") or []),
        "completion_released": bool(completion.get("released")),
        "confidence_brackets": any(
            str(item.get("rendered") or "").strip()
            for item in ix_render.get("certainties") or []
        ),
    }


#: The PA-facing chat identity, opaque and identical in every run.
#:
#: `pa/agent/context.py:170-171` renders `Channel:` / `Chat ID:` straight into
#: the runtime-context block of the prompt, and the caller used to pass the
#: run-derived session key (`simulator/loop.py`: `chat_id=self.session_key`,
#: i.e. `<run_id>_main`). A run id such as `ixval_proactive_low_002_main` names
#: the setting under test, and the PA read it as one: two of three proactive
#: runs justified their choice with "the channel name … indicates this turn
#: evaluates the Low setting". Nothing the model can see may carry the run id,
#: so the harness pins the pair here for every PA call it makes.
#:
#: The run id keeps living on the harness side only — ledger, run directory,
#: transcript, and the on-disk `pa_session_key` (`<run_id>_main`) that
#: checkpoint / resume addresses sessions by. `session_key` is never rendered
#: into a prompt; only `chat_id` was.
PA_CHANNEL = "benchmark"
PA_CHAT_ID = "main"


class RecordingPA:
    """Wrap a PA loop and record benchmark-facing input payloads.

    This is the object the simulator holds as `pa_loop`, so it is also where
    the harness translates the PA's IX face into the plain keys the simulator
    reads. `simulator/` never imports `pa/` and never branches on `ix_render`.

    It is also the one seam every live benchmark PA call passes through, so it
    is where the model-visible chat identity is pinned to `PA_CHANNEL` /
    `PA_CHAT_ID` regardless of what the caller asked for.
    """

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.received: list[dict] = []
        self.ix_render_records: list[dict] = []

    async def process_direct_message(
        self,
        user_message: str,
        session_key: str = "sim:default",
        **kwargs: Any,
    ) -> Any:
        metadata = kwargs.get("metadata") or {}
        self.received.append({
            "content": user_message,
            "session_key": session_key,
            "metadata": metadata,
        })
        return self._translate(
            await self.inner.process_direct_message(
                user_message=user_message,
                session_key=session_key,
                **{**kwargs, "channel": PA_CHANNEL, "chat_id": PA_CHAT_ID},
            ),
            kind="direct",
            call_metadata=metadata,
            session_key=session_key,
        )

    async def process_environment_event(self, **kwargs: Any) -> Any:
        metadata = kwargs.get("metadata") or {}
        self.received.append({
            "content": kwargs.get("detail") or "",
            "session_key": kwargs.get("session_key") or "sim:default",
            "metadata": metadata,
            "source": "environment",
        })
        return self._translate(
            await self.inner.process_environment_event(
                **{**kwargs, "channel": PA_CHANNEL, "chat_id": PA_CHAT_ID},
            ),
            kind="environment",
            call_metadata=metadata,
            session_key=kwargs.get("session_key") or "sim:default",
        )

    def close_ix_session(
        self, session_key: str, script_session_id: str | None = None
    ) -> list[tuple[str, str]]:
        """Pass the step-boundary close through to the wrapped PA (§4.2)."""
        return self.inner.close_ix_session(session_key, script_session_id)

    def _triggered_skills(
        self, session_key: str, call_metadata: Mapping[str, Any]
    ) -> list[str]:
        read = getattr(self.inner, "triggered_ix_skills", None)
        if read is None:
            return []
        return read(session_key, call_metadata.get("script_session_id"))

    def _translate(
        self,
        outbound: Any,
        *,
        kind: str,
        call_metadata: Mapping[str, Any],
        session_key: str,
    ) -> Any:
        """Capture the turn's `ix_render`, then add the simulator-facing keys.

        The capture is why this wrapper records at all for the IX face: the
        simulator emits `pa_turn` **without** the PA's metadata (it must not
        read `ix_render`), so this is the only place the block still exists on
        its way past. One record per PA call, in call order — including calls
        that rendered nothing, so position still lines up with `pa_turn`.
        """
        metadata = getattr(outbound, "metadata", None)
        metadata = metadata if isinstance(metadata, dict) else {}
        ix_render = metadata.get("ix_render")
        self.ix_render_records.append({
            "kind": kind,
            "turn_id": call_metadata.get("turn_id"),
            "chat_id": getattr(outbound, "chat_id", None),
            "ix_render": ix_render or None,
        })
        # Which attributes a runtime trigger activated in this session, rather
        # than a beat's `active_skills`. Written on **every** PA turn, `[]` when
        # nothing has fired, and sticky once it has: `capability_boundary` is
        # listed from the turn a tool reported a `capability_limit` onward, so
        # the simulator can tell that the attribute is live on a beat whose
        # script never named it (audit §2, acc_026 — the limit fired two beats
        # before the beat that listed the skill). Like `has_pending_interaction`
        # this carries no setting, no text and no ids: only which attribute.
        metadata["triggered_skills"] = self._triggered_skills(
            session_key, call_metadata
        )
        if not ix_render:
            return outbound
        metadata.update(outreach_keys_from_ix_render(ix_render))
        held_items = held_items_from_ix_render(ix_render)
        if held_items:
            metadata["held_items"] = held_items
        # Pacing only, and always written: a hold with nothing consent-able
        # still keeps the beat open, and `held_items` cannot say so.
        metadata["has_pending_interaction"] = has_pending_interaction_from_ix_render(
            ix_render
        )
        # The actor's record of the delivered reply (owner 2026-09-17): three
        # counts/flags off what went out, never what was held.
        metadata["reply_facts"] = reply_facts_from_ix_render(ix_render)
        return outbound


# ---------------------------------------------------------------------------
# IX wrapper seams (plan §3.8 contract 10, §4.2 contract 7)
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CATALOG_RELATIVE = Path("data/state_tools/catalog.yaml")


class OpenHoldEpisodeError(RuntimeError):
    """A step ended with a hold episode still open (§4.2, contract 7).

    The step failed and **no checkpoint may be published**: a cut point that
    carried an open hold would restore into a session whose PA is waiting on a
    confirmation nobody will ever give.

    "Waiting on a confirmation" is the whole of it, and owner ruling 2026-09-18
    holds the wrapper to it: what makes an episode open is an item the user
    still owes an answer on -- an autonomy ask, a held question, an undelivered
    batch, a call waiting for its release. A *delivered* `handoff_message` is
    the opposite: the PA has handed control back and is waiting for nothing, so
    `capability_boundary` closes on that turn (`pa/agent/ix/render.py`,
    `_next_hold`) instead of being carried into a turn a compressed session
    never has. Nothing else changed: any episode that really is waiting still
    fails the step here, on the mainline. A probe step is the one place a
    still-open episode is recorded rather than raised -- it runs in its own
    branch and publishes no checkpoint (`harness/orchestrator.py`,
    `_run_probe_in_branch`), so the open hold lands in its `meta.yaml`.
    """

    def __init__(self, step_id: str, episodes: list[tuple[str, str]]) -> None:
        self.step_id = step_id
        self.episodes = episodes
        listed = ", ".join(f"{attribute}={episode}" for attribute, episode in episodes)
        super().__init__(
            "step ended with an open IX hold episode, so it failed and no "
            f"checkpoint was published: step_id={step_id!r} open={listed}"
        )


def _catalog_root(project_root: Path | str) -> Path:
    """Where the world-toolbox catalog lives.

    A synthetic run directory (tests, scratch harnesses) has no catalog of its
    own; the coverage check still has to run against the real toolbox, so fall
    back to the repository's own copy rather than skipping the check.
    """
    candidate = Path(project_root)
    if (candidate / _CATALOG_RELATIVE).exists():
        return candidate
    return _REPO_ROOT


def assert_benchmark_action_type_coverage(
    project_root: Path | str,
    *,
    overrides: Mapping[str, Any] | None = None,
) -> list[str]:
    """Contract 10: every standing task tool must have a declared action type.

    Runs **once at benchmark start**, before any step touches the world. The
    tool surface is read from the world-toolbox catalog rather than from a live
    MCP connection, so an unclassified tool fails the run up front instead of
    at the turn that first calls it — where `tool_action_type()`'s fail-closed
    floor would silently gate it as `external_action` (§3.8).
    """
    tools = world_toolbox_tools(_catalog_root(project_root))
    assert_action_type_coverage(tools, overrides)
    return tools


def ix_render_records_from_loop(loop: Any) -> list[dict]:
    """The per-call `ix_render` records a finished session produced.

    The session loop the executor holds does not own the PA; the `RecordingPA`
    the simulator talks through does. Two shallow, documented hops find it —
    the simulator's own `pa_loop`, or the recorder a wrapping session loop
    kept — and anything else yields nothing to merge rather than an error, so
    a loop that never used a `RecordingPA` (fakes, forced events) is fine.
    """
    for candidate in (loop, getattr(loop, "pa_recorder", None), getattr(loop, "pa_loop", None)):
        records = getattr(candidate, "ix_render_records", None)
        if isinstance(records, list):
            return records
    return []
