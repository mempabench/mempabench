"""OpenAI-compatible provider for all non-Anthropic LLM APIs."""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import string
import sys
import time
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx
import json_repair
from openai import APITimeoutError, AsyncOpenAI

from pa.providers.base import LLMProvider, LLMResponse, ServedBy, ToolCallRequest
from pa.providers.request_policy import (
    DEFAULT_MODEL_POLICY_CATALOG,
    ModelRequestPolicy,
    resolve_model_request_policy,
)

if TYPE_CHECKING:
    from harness.provider_avoidance import ProviderAvoidance
    from pa.providers.registry import ProviderSpec

#: Wall-clock ceiling on one provider request. acc_039 sat nine
#: minutes on its first OpenRouter call with nothing on stderr: the SDK default
#: is a 600 s timeout with two internal retries, so a stalled connection could
#: hold a run for half an hour before anything surfaced. The SDK retries are
#: off (`max_retries=0` below) and the total is 600 s, long enough for a PA or
#: File memory consolidation call to return the plan's whole `pa_max_tokens`.
#: `MEMPA_LLM_REQUEST_TIMEOUT_S` overrides the total; the connect leg stays at
#: 20 s. Duplicated in simulator/providers/openai_compat_provider.py on
#: purpose (pa/ and simulator/ never share modules); the simulated user's
#: shorter replies keep its 180 s default there.
_DEFAULT_REQUEST_TIMEOUT_S = 600.0
_CONNECT_TIMEOUT_S = 20.0
_REQUEST_TIMEOUT_ENV = "MEMPA_LLM_REQUEST_TIMEOUT_S"


def _request_timeout() -> httpx.Timeout:
    """The per-request deadline, read fresh from the environment."""
    total = float(os.environ.get(_REQUEST_TIMEOUT_ENV) or _DEFAULT_REQUEST_TIMEOUT_S)
    return httpx.Timeout(total, connect=_CONNECT_TIMEOUT_S)

_ALLOWED_MSG_KEYS = frozenset({
    "role", "content", "tool_calls", "tool_call_id", "name",
    "reasoning_content", "extra_content",
})
#: Provider-side reasoning keys, read in this order for **audit capture only**:
#: `reasoning_content` is the Kimi /
#: DeepSeek-R1 name, `reasoning` is what OpenRouter normalises to, and
#: `reasoning_details` is the structured variant. None of it is ever rendered.
PROVIDER_REASONING_KEYS: tuple[str, str, str] = (
    "reasoning_content", "reasoning", "reasoning_details",
)
_ALNUM = string.ascii_letters + string.digits

_STANDARD_TC_KEYS = frozenset({"id", "type", "index", "function"})
_STANDARD_FN_KEYS = frozenset({"name", "arguments"})
_DEFAULT_OPENROUTER_HEADERS = {
    "HTTP-Referer": "https://github.com/HKUDS/nanobot",
    "X-OpenRouter-Title": "nanobot",
    "X-OpenRouter-Categories": "cli-agent,personal-agent",
}


def _short_tool_id() -> str:
    """9-char alphanumeric ID compatible with all providers (incl. Mistral)."""
    return "".join(secrets.choice(_ALNUM) for _ in range(9))


#: `deepseek/deepseek-v4-flash` on OpenRouter sometimes answers with its own
#: native tool-call syntax typed into `content` instead of structured
#: `tool_calls`. acc_006: six of eleven PA turns arrived that way, five
#: of them shipped the empty-reply template with no task call at all, and the
#: session's writes fell from 8 to 1. The block the model writes is
#:
#:     <｜DSML｜tool_calls>
#:     <｜DSML｜invoke name="mcp_state_document_read">
#:     <｜DSML｜parameter name="path" string="true">documents/x.md</｜DSML｜parameter>
#:     </｜DSML｜invoke>
#:     </｜DSML｜tool_calls>
#:
#: `string="true"` says the element's text **is** the value, verbatim, newlines
#: and all — `document_update`'s `old_string` / `new_string` arrive as multi-line
#: literals whose leading newline is load-bearing, so nothing here strips.
#: `string="false"` says the text is JSON (`anchored_replacements` arrives as a
#: JSON array). The bar is U+FF5C, full width, which is what the model really
#: emits; the ASCII pipe and the `<|DSML|>tool_calls>` spelling are matched too,
#: same tolerance as the renderer's stripper.
_DSML_MARK = r"[|｜]\s*DSML\s*[|｜]>?"
_DSML_BLOCK_RE = re.compile(
    rf"<{_DSML_MARK}\s*tool_calls>(?P<body>.*?)</{_DSML_MARK}\s*tool_calls>",
    re.DOTALL,
)
_DSML_INVOKE_RE = re.compile(
    rf'<{_DSML_MARK}\s*invoke\s+name="(?P<name>[^"]*)"\s*>'
    rf"(?P<body>.*?)</{_DSML_MARK}\s*invoke>",
    re.DOTALL,
)
_DSML_PARAM_RE = re.compile(
    rf'<{_DSML_MARK}\s*parameter\s+name="(?P<name>[^"]*)"'
    rf'(?:\s+string="(?P<string>true|false)")?\s*>'
    rf"(?P<value>.*?)</{_DSML_MARK}\s*parameter>",
    re.DOTALL,
)
_BLANK_RUN_RE = re.compile(r"\n{3,}")


def _parse_dsml_block(body: str) -> list[tuple[str, dict[str, Any]]] | None:
    """The `(name, arguments)` of every invoke in one block, or None if any of it is malformed.

    All or nothing per block. acc_001 t32's block was truncated mid-parameter,
    and half an argument list executed against the user's documents is worse
    than a block nobody runs: an unparsable block goes back to the renderer's
    stripper, which drops it and records `ix_markup_stripped`.
    """
    invokes = list(_DSML_INVOKE_RE.finditer(body))
    if not invokes:
        return None
    if "DSML" in _DSML_INVOKE_RE.sub("", body):
        # Markup the invoke pattern could not close — an unterminated invoke
        # sitting next to a complete one.
        return None
    parsed: list[tuple[str, dict[str, Any]]] = []
    for invoke in invokes:
        name = invoke.group("name").strip()
        if not name:
            return None
        if "DSML" in _DSML_PARAM_RE.sub("", invoke.group("body")):
            return None
        arguments: dict[str, Any] = {}
        for param in _DSML_PARAM_RE.finditer(invoke.group("body")):
            value: Any = param.group("value")
            if param.group("string") == "false":
                try:
                    value = json.loads(value)
                except ValueError:
                    return None
            arguments[param.group("name")] = value
        parsed.append((name, arguments))
    return parsed


def convert_dsml_tool_calls(
    content: str | None,
) -> tuple[str | None, list[ToolCallRequest]]:
    """Lift the model's typed-out tool calls out of the prose and into real calls.

    Returns `(content without the converted blocks, the calls they carried)`.
    Ids are minted `dsml_<n>` in the order they appear, so the same response
    always yields the same ids.

    A block that does not parse is left in `content` exactly as it arrived: the
    IX renderer's existing filter strips it and records `ix_markup_stripped`,
    which is the audit row for a call the PA meant to make and that nobody can
    reconstruct. Doing it here as well would take the block away before the
    record was written.
    """
    if not content or "DSML" not in content:
        return content, []
    calls: list[ToolCallRequest] = []

    def _cut(match: re.Match[str]) -> str:
        parsed = _parse_dsml_block(match.group("body"))
        if parsed is None:
            return match.group(0)
        for name, arguments in parsed:
            calls.append(
                ToolCallRequest(id=f"dsml_{len(calls) + 1}", name=name, arguments=arguments)
            )
        return ""

    kept = _DSML_BLOCK_RE.sub(_cut, content)
    if not calls:
        return content, []
    return _BLANK_RUN_RE.sub("\n\n", kept).strip() or None, calls


def _get(obj: Any, key: str) -> Any:
    """Get a value from dict or object attribute, returning None if absent."""
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _coerce_dict(value: Any) -> dict[str, Any] | None:
    """Try to coerce *value* to a dict; return None if not possible or empty."""
    if value is None:
        return None
    if isinstance(value, dict):
        return value if value else None
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        if isinstance(dumped, dict) and dumped:
            return dumped
    return None


def _extract_tc_extras(tool_call: Any) -> tuple[
    dict[str, Any] | None,
    dict[str, Any] | None,
    dict[str, Any] | None,
]:
    """Extract (extra_content, provider_specific_fields, fn_provider_specific_fields).

    Works for both SDK objects and dicts.  Captures Gemini ``extra_content``
    verbatim and any non-standard keys on the tool-call / function.
    """
    extra_content = _coerce_dict(_get(tool_call, "extra_content"))

    tool_call_map = _coerce_dict(tool_call)
    provider_fields = None
    function_provider_fields = None
    if tool_call_map is not None:
        leftover = {k: v for k, v in tool_call_map.items()
                    if k not in _STANDARD_TC_KEYS and k != "extra_content" and v is not None}
        if leftover:
            provider_fields = leftover
        function_payload = _coerce_dict(tool_call_map.get("function"))
        if function_payload is not None:
            fn_leftover = {k: v for k, v in function_payload.items()
                          if k not in _STANDARD_FN_KEYS and v is not None}
            if fn_leftover:
                function_provider_fields = fn_leftover
    else:
        provider_fields = _coerce_dict(_get(tool_call, "provider_specific_fields"))
        function_object = _get(tool_call, "function")
        if function_object is not None:
            function_provider_fields = _coerce_dict(
                _get(function_object, "provider_specific_fields")
            )

    return extra_content, provider_fields, function_provider_fields


def _uses_openrouter_attribution(spec: "ProviderSpec | None", api_base: str | None) -> bool:
    """Apply Nanobot attribution headers to OpenRouter requests by default."""
    if spec and spec.name == "openrouter":
        return True
    return bool(api_base and "openrouter" in api_base.lower())



#: OpenRouter's refusal when the request's ignore list covers every upstream
#: of the model (live): HTTP 404, "All providers have been
#: ignored. ...", with this routing step in `error.metadata`.
_IGNORED_ALL_ROUTING_STEP = "Filter by Ignored Providers"


def _request_provider_ignore(request_payload: dict[str, Any]) -> list[str]:
    """The `provider.ignore` list the request carries."""
    routing = (request_payload.get("extra_body") or {}).get("provider") or {}
    return list(routing.get("ignore") or [])


def _error_upstream_provider(error: Exception) -> str | None:
    """The upstream an OpenRouter error body names (`error.metadata.provider_name`).

    The SDK hands over the body's `error` object. A timeout, a connection
    failure or an error OpenRouter raised itself names none.
    """
    body = getattr(error, "body", None)
    metadata = body.get("metadata") if isinstance(body, dict) else None
    name = metadata.get("provider_name") if isinstance(metadata, dict) else None
    return name if isinstance(name, str) and name else None


def _empty_reply(response: LLMResponse) -> bool:
    """No non-blank text content and no tool call carrying arguments.

    A tool call whose arguments are missing, blank or `{}` carries none.
    """
    return not (response.content or "").strip() and not any(
        call.arguments for call in response.tool_calls
    )


def _upstream_failure_kind(response: LLMResponse) -> str | None:
    """The provider avoidance failure kind `response` is on arrival, or None.

    `provider_error` for an error response, `provider_truncation` for a reply
    cut off at the token limit with no content and no tool arguments in it.
    """
    if response.finish_reason == "error":
        return "provider_error"
    if response.finish_reason == "length" and _empty_reply(response):
        return "provider_truncation"
    return None


def _log_openrouter_call(
    model: str,
    started: float,
    response: LLMResponse,
    cost: Any,
    task_key: str | None = None,
    upstream_provider: str | None = None,
    provider_ignore: list[str] | None = None,
) -> None:
    """One unbuffered stderr line per provider call, for live run monitoring.

    Duplicated in simulator/providers/openai_compat_provider.py on purpose: pa/
    and simulator/ never share modules.
    """
    usage = response.usage or {}
    line = (
        f"openrouter_call model={model} seconds={time.monotonic() - started:.0f} "
        f"prompt_tokens={usage.get('prompt_tokens')} "
        f"completion_tokens={usage.get('completion_tokens')}"
    )
    if cost is not None:
        line += f" cost_usd={cost}"
    # The caller's LLMTaskIdentity, when it set one: without it a memory-side
    # extraction line cannot be attributed to the turn that caused it.
    if task_key:
        line += f" task={task_key}"
    # One model id can be served by several upstreams whose forced-tool_choice
    # behaviour differs (`z-ai/glm-5.3-flash`), so the line names
    # the one that answered. Omitted entirely when the gateway does not say.
    if upstream_provider:
        line += f" upstream_provider={upstream_provider}"
    # The upstreams the request told OpenRouter to skip (provider avoidance).
    if provider_ignore:
        line += f" provider_ignore={','.join(provider_ignore)}"
    if response.dsml_converted:
        line += f" dsml_tool_calls={response.dsml_converted}"
    # A run on gemini-3.8-flash cut 7/13 self-report calls at completion_tokens=4092
    # and the log could not say `length`.
    line += f" finish_reason={response.finish_reason}"
    print(line, file=sys.stderr, flush=True)


class OpenAICompatProvider(LLMProvider):
    """Unified provider for all OpenAI-compatible APIs.

    Receives a resolved ``ProviderSpec`` from the caller — no internal
    registry lookups needed.
    """

    def __init__(
        self,
        api_key: str | None = None,
        api_base: str | None = None,
        default_model: str = "gpt-4o",
        extra_headers: dict[str, str] | None = None,
        spec: ProviderSpec | None = None,
        request_policy_catalog_path: Path | str = DEFAULT_MODEL_POLICY_CATALOG,
        extra_body: dict[str, Any] | None = None,
        provider_avoidance: ProviderAvoidance | None = None,
    ):
        super().__init__(api_key, api_base)
        self.default_model = default_model
        self.extra_headers = extra_headers or {}
        self._spec = spec
        self._request_policy_catalog_path = request_policy_catalog_path
        #: Top-level request-body keys merged in verbatim: OpenRouter upstream
        #: routing, set by `harness/pa_factory.py` from `provider_routing` in
        #: configs/model_request_policies.yaml and, on probe steps, from
        #: `providers.probe_phase` in configs/run_config.yaml. `None` sends the
        #: payload the request policy builds. Not a `ModelRequestPolicy` field,
        #: so it never changes a policy snapshot.
        self.extra_body = extra_body
        #: The run's OpenRouter upstream avoidance (`harness/provider_avoidance.py`),
        #: set by `harness/pa_factory.py`: failures are counted against the
        #: upstream that served them, and each request skips the upstreams it
        #: lists for the model in `provider.ignore`, merged into `extra_body`.
        #: Neither happens while `extra_body` pins the routing. None: off.
        self.provider_avoidance = provider_avoidance
        #: Logging only: when set, every PA
        #: call's raw upstream response (or raw provider error body) is appended here. No auth data.
        self.raw_response_log: Path | None = None
        #: Upstream provider ids (OpenRouter's `provider`) of the calls made
        #: since the last `reset_upstream_providers()`, in order. The PA loop
        #: resets it per turn and reads it when a turn fails the IX contract:
        #: `z-ai/glm-5.3-flash` is served by several upstreams whose
        #: forced-`tool_choice` behaviour differs, and nothing else records
        #: which one answered. Ids only — never a response body.
        self._upstream_providers: list[str] = []

        if api_key and spec and spec.env_key:
            self._setup_env(api_key, api_base)

        effective_base = api_base or (spec.default_api_base if spec else None) or None
        default_headers = {"x-session-affinity": uuid.uuid4().hex}
        if _uses_openrouter_attribution(spec, effective_base):
            default_headers.update(_DEFAULT_OPENROUTER_HEADERS)
        if extra_headers:
            default_headers.update(extra_headers)

        self._client = AsyncOpenAI(
            api_key=api_key or "no-key",
            base_url=effective_base,
            default_headers=default_headers,
            timeout=_request_timeout(),
            # `chat_with_retry` is the only retry layer we want; the SDK's own
            # two retries would multiply the deadline above by three.
            max_retries=0,
        )

    def supports_forced_tool_choice(self, model: str | None = None) -> bool:
        """Read the answer off this model's request policy, not off a guess."""
        policy = self._request_policy(model or self.default_model)
        return bool(policy.supports_tools and policy.supports_forced_tool_choice)

    def _request_policy(self, model_name: str) -> ModelRequestPolicy:
        provider_name = self._spec.name if self._spec else "custom"
        return resolve_model_request_policy(
            provider_name,
            model_name,
            catalog_path=self._request_policy_catalog_path,
        )

    def _setup_env(self, api_key: str, api_base: str | None) -> None:
        """Set environment variables based on provider spec."""
        spec = self._spec
        if not spec or not spec.env_key:
            return
        if spec.is_gateway:
            os.environ[spec.env_key] = api_key
        else:
            os.environ.setdefault(spec.env_key, api_key)
        effective_base = api_base or spec.default_api_base
        for env_name, env_val in spec.env_extras:
            resolved = env_val.replace("{api_key}", api_key).replace("{api_base}", effective_base)
            os.environ.setdefault(env_name, resolved)

    @staticmethod
    def _apply_cache_control(
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None]:
        """Inject cache_control markers for prompt caching."""
        cache_marker = {"type": "ephemeral"}
        new_messages = list(messages)

        def _mark(msg: dict[str, Any]) -> dict[str, Any]:
            content = msg.get("content")
            if isinstance(content, str):
                return {**msg, "content": [
                    {"type": "text", "text": content, "cache_control": cache_marker},
                ]}
            if isinstance(content, list) and content:
                nc = list(content)
                nc[-1] = {**nc[-1], "cache_control": cache_marker}
                return {**msg, "content": nc}
            return msg

        if new_messages and new_messages[0].get("role") == "system":
            new_messages[0] = _mark(new_messages[0])
        if len(new_messages) >= 3:
            new_messages[-2] = _mark(new_messages[-2])

        new_tools = tools
        if tools:
            new_tools = list(tools)
            new_tools[-1] = {**new_tools[-1], "cache_control": cache_marker}
        return new_messages, new_tools

    @staticmethod
    def _normalize_tool_call_id(tool_call_id: Any) -> Any:
        """Normalize to a provider-safe 9-char alphanumeric form."""
        if not isinstance(tool_call_id, str):
            return tool_call_id
        if len(tool_call_id) == 9 and tool_call_id.isalnum():
            return tool_call_id
        return hashlib.sha1(tool_call_id.encode()).hexdigest()[:9]

    def _sanitize_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Strip non-standard keys, normalize tool_call IDs."""
        sanitized = LLMProvider._sanitize_request_messages(messages, _ALLOWED_MSG_KEYS)
        id_map: dict[str, str] = {}

        def map_id(value: Any) -> Any:
            if not isinstance(value, str):
                return value
            return id_map.setdefault(value, self._normalize_tool_call_id(value))

        for clean in sanitized:
            if isinstance(clean.get("tool_calls"), list):
                normalized = []
                for tool_call in clean["tool_calls"]:
                    if not isinstance(tool_call, dict):
                        normalized.append(tool_call)
                        continue
                    clean_tool_call = dict(tool_call)
                    clean_tool_call["id"] = map_id(clean_tool_call.get("id"))
                    normalized.append(clean_tool_call)
                clean["tool_calls"] = normalized
            if "tool_call_id" in clean and clean["tool_call_id"]:
                clean["tool_call_id"] = map_id(clean["tool_call_id"])
        return sanitized

    # ------------------------------------------------------------------
    # Build kwargs
    # ------------------------------------------------------------------

    def _build_kwargs(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        model: str | None,
        max_tokens: int,
        temperature: float,
        reasoning_effort: str | None,
        tool_choice: str | dict[str, Any] | None,
    ) -> dict[str, Any]:
        model_name = model or self.default_model
        spec = self._spec
        request_policy = self._request_policy(model_name)

        if spec and spec.supports_prompt_caching:
            messages, tools = self._apply_cache_control(messages, tools)

        if spec and spec.strip_model_prefix:
            model_name = model_name.split("/")[-1]

        request_payload: dict[str, Any] = {
            "model": model_name,
            "messages": self._sanitize_messages(self._sanitize_empty_content(messages)),
        }

        if request_policy.temperature == "send":
            request_payload["temperature"] = temperature

        if request_policy.completion_tokens_parameter == "max_completion_tokens":
            request_payload["max_completion_tokens"] = max(1, max_tokens)
        else:
            request_payload["max_tokens"] = max(1, max_tokens)

        if reasoning_effort and request_policy.reasoning_effort == "send":
            request_payload["reasoning_effort"] = reasoning_effort

        if tools:
            if not request_policy.supports_tools:
                raise ValueError(
                    "Model request policy does not support tools: "
                    f"provider_name={request_policy.provider_name!r} "
                    f"model={model_name!r}"
                )
            if (
                isinstance(tool_choice, dict)
                and not request_policy.supports_forced_tool_choice
            ):
                raise ValueError(
                    "Model request policy does not support forced tool_choice: "
                    f"provider_name={request_policy.provider_name!r} "
                    f"model={model_name!r}"
                )
            request_payload["tools"] = tools
            request_payload["tool_choice"] = tool_choice or "auto"

        extra_body = self._request_extra_body(request_payload["model"])
        if extra_body:
            # The SDK's own passthrough: `AsyncCompletions.create` has no
            # `**kwargs`, so a non-OpenAI body field such as OpenRouter's
            # `provider` must ride here or the call raises TypeError before any
            # request is sent. The SDK merges it into the JSON body.
            request_payload["extra_body"] = extra_body

        return request_payload

    # ------------------------------------------------------------------
    # Provider avoidance
    # ------------------------------------------------------------------

    def _routing_pinned(self) -> bool:
        """`order` with `allow_fallbacks: false`, as HY4-Preview on SiliconFlow."""
        routing = (self.extra_body or {}).get("provider") or {}
        return bool(routing.get("order")) and routing.get("allow_fallbacks") is False

    def _avoidance_applies(self) -> bool:
        return (
            self.provider_avoidance is not None
            and self.provider_avoidance.enabled
            and not self._routing_pinned()
        )

    def _request_extra_body(self, model: str) -> dict[str, Any] | None:
        """`extra_body`, with the run's avoided upstreams for `model` added to `provider.ignore`."""
        if not self._avoidance_applies():
            return self.extra_body
        avoided = self.provider_avoidance.ignored_providers(model)
        if not avoided:
            return self.extra_body
        routing = dict((self.extra_body or {}).get("provider") or {})
        routing["ignore"] = list(dict.fromkeys([*(routing.get("ignore") or []), *avoided]))
        return {**(self.extra_body or {}), "provider": routing}

    def _count_upstream_failure(self, response: LLMResponse, kind: str, failure: str) -> None:
        if response.served_by is None or not self._avoidance_applies():
            return
        self.provider_avoidance.record_failure(
            response.served_by.model, response.served_by.provider, kind=kind, failure=failure,
        )

    def _count_arrival_failure(self, response: LLMResponse) -> None:
        """Count a response that is a provider failure as it arrives."""
        kind = _upstream_failure_kind(response)
        if kind == "provider_error":
            self._count_upstream_failure(response, kind, response.error_text)
        elif kind == "provider_truncation":
            self._count_upstream_failure(
                response, kind,
                "finish_reason=length with no content and no tool arguments "
                f"completion_tokens={(response.usage or {}).get('completion_tokens')}",
            )

    def record_contract_failure(self, response: LLMResponse, reason: str) -> None:
        """Charge the upstream that served `response` for its caller's
        rejection, when `response` is empty.

        A rejected response with text or tool arguments is the model's own
        failure, the same on every upstream, and is not counted. A response
        already counted as a provider failure when it arrived is not counted
        again.
        """
        if _upstream_failure_kind(response) is None and _empty_reply(response):
            self._count_upstream_failure(response, "contract_rejected", reason)

    def _no_upstream_left(self, error: Exception) -> bool:
        """OpenRouter refused the request because its ignore list left the model no upstream."""
        body = getattr(error, "body", None)
        metadata = body.get("metadata") if isinstance(body, dict) else None
        return (
            self._avoidance_applies()
            and getattr(error, "status_code", None) == 404
            and isinstance(metadata, dict)
            and metadata.get("failed_routing_step") == _IGNORED_ALL_ROUTING_STEP
        )

    def _reset_avoidance(self, request_payload: dict[str, Any]) -> dict[str, Any]:
        """Clear the run's avoidance for the request's model; return the same
        request without the avoided upstreams."""
        self.provider_avoidance.reset(request_payload["model"])
        resend = {key: value for key, value in request_payload.items() if key != "extra_body"}
        if self.extra_body:
            resend["extra_body"] = self.extra_body
        return resend

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _maybe_mapping(value: Any) -> dict[str, Any] | None:
        if isinstance(value, dict):
            return value
        model_dump = getattr(value, "model_dump", None)
        if callable(model_dump):
            dumped = model_dump()
            if isinstance(dumped, dict):
                return dumped
        return None

    @classmethod
    def _extract_provider_reasoning(cls, message: Any) -> str | None:
        """Audit-only capture of provider-side reasoning.

        This is **never rendered** and never reaches
        `public_turn` / `runtime_history` / `memory_projection` — it exists only so
        `AuditTrace.provider_reasoning` can record whether the provider returned any.
        OpenRouter normalises the field to `reasoning` (not `reasoning_content`), and
        some providers return a structured `reasoning_details` list; read all three.
        """
        message_map = cls._maybe_mapping(message)
        for key in PROVIDER_REASONING_KEYS:
            value = (
                message_map.get(key)
                if message_map is not None
                else getattr(message, key, None)
            )
            if isinstance(value, str):
                if value:
                    return value
            elif isinstance(value, (list, dict)) and value:
                return json.dumps(value, ensure_ascii=False, default=str)
        return None

    @classmethod
    def _extract_text_content(cls, value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            parts: list[str] = []
            for item in value:
                item_map = cls._maybe_mapping(item)
                if item_map:
                    text = item_map.get("text")
                    if isinstance(text, str):
                        parts.append(text)
                        continue
                text = getattr(item, "text", None)
                if isinstance(text, str):
                    parts.append(text)
                    continue
                if isinstance(item, str):
                    parts.append(item)
            return "".join(parts) or None
        return str(value)

    @classmethod
    def _extract_usage(cls, provider_response: Any) -> dict[str, int]:
        usage_obj = None
        response_map = cls._maybe_mapping(provider_response)
        if response_map is not None:
            usage_obj = response_map.get("usage")
        elif hasattr(provider_response, "usage") and provider_response.usage:
            usage_obj = provider_response.usage

        usage_map = cls._maybe_mapping(usage_obj)
        if usage_map is not None:
            usage = {
                "prompt_tokens": int(usage_map.get("prompt_tokens") or 0),
                "completion_tokens": int(usage_map.get("completion_tokens") or 0),
                "total_tokens": int(usage_map.get("total_tokens") or 0),
            }
            # A reasoning model can spend the whole completion budget thinking
            # and return empty content, and `completion_tokens` alone cannot
            # say that it did. Carried only when the gateway reports it, so a
            # response without the block keeps exactly the three old keys.
            details = cls._maybe_mapping(usage_map.get("completion_tokens_details"))
            reasoning_tokens = (details or {}).get("reasoning_tokens")
            if reasoning_tokens is not None:
                usage["reasoning_tokens"] = int(reasoning_tokens)
            return usage

        if usage_obj:
            return {
                "prompt_tokens": getattr(usage_obj, "prompt_tokens", 0) or 0,
                "completion_tokens": getattr(usage_obj, "completion_tokens", 0) or 0,
                "total_tokens": getattr(usage_obj, "total_tokens", 0) or 0,
            }
        return {}

    @classmethod
    def _extract_cost(cls, provider_response: Any) -> Any:
        """OpenRouter reports credit spend in usage.cost; other gateways do not."""
        response_map = cls._maybe_mapping(provider_response) or {}
        usage_map = cls._maybe_mapping(response_map.get("usage")) or {}
        return usage_map.get("cost")

    @classmethod
    def _extract_upstream_provider(cls, provider_response: Any) -> str | None:
        """Which upstream actually served the call (OpenRouter's `provider`).

        One model id is not one implementation: an inspection of
        `z-ai/glm-5.3-flash` sent byte-identical forced-`tool_choice` requests
        and got the pinned tool from `Wafer` / `Cloudflare` / `Together` but
        never from `Phala`, which either ignored the pin or dropped the schema's
        required field. The log line and the failed-turn record could not say
        which upstream answered, so a run's failures could not be attributed.
        Other gateways omit the key and get `None`.
        """
        response_map = cls._maybe_mapping(provider_response) or {}
        value = response_map.get("provider")
        return value if isinstance(value, str) and value else None

    @staticmethod
    def _provider_error_detail(
        response_map: dict[str, Any],
        choice_map: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "response_id": response_map.get("id"),
            "model": response_map.get("model"),
            "response_error": response_map.get("error"),
            "raw_choice": choice_map,
        }

    def _parse(self, provider_response: Any) -> LLMResponse:
        if isinstance(provider_response, str):
            return LLMResponse(content=provider_response, finish_reason="stop")

        response_map = self._maybe_mapping(provider_response)
        if response_map is not None:
            choices = response_map.get("choices") or []
            if not choices:
                content = self._extract_text_content(
                    response_map.get("content") or response_map.get("output_text")
                )
                if content is not None:
                    return LLMResponse(
                        content=content,
                        finish_reason=str(response_map.get("finish_reason") or "stop"),
                        usage=self._extract_usage(response_map),
                    )
                return LLMResponse(content="Error: API returned empty choices.", finish_reason="error")

            choice0 = self._maybe_mapping(choices[0]) or {}
            msg0 = self._maybe_mapping(choice0.get("message")) or {}
            content = self._extract_text_content(msg0.get("content"))
            finish_reason = str(choice0.get("finish_reason") or "stop")

            raw_tool_calls: list[Any] = []
            reasoning_content = self._extract_provider_reasoning(msg0)
            for choice in choices:
                choice_map = self._maybe_mapping(choice) or {}
                message_map = self._maybe_mapping(choice_map.get("message")) or {}
                tool_calls = message_map.get("tool_calls")
                if isinstance(tool_calls, list) and tool_calls:
                    raw_tool_calls.extend(tool_calls)
                    if choice_map.get("finish_reason") in ("tool_calls", "stop"):
                        finish_reason = str(choice_map["finish_reason"])
                if not content:
                    content = self._extract_text_content(message_map.get("content"))
                if not reasoning_content:
                    reasoning_content = self._extract_provider_reasoning(message_map)

            parsed_tool_calls = []
            for tool_call in raw_tool_calls:
                tool_call_map = self._maybe_mapping(tool_call) or {}
                function_payload = self._maybe_mapping(tool_call_map.get("function")) or {}
                arguments = function_payload.get("arguments", {})
                if isinstance(arguments, str):
                    arguments = json_repair.loads(arguments)
                extra_content, provider_fields, function_provider_fields = (
                    _extract_tc_extras(tool_call)
                )
                parsed_tool_calls.append(ToolCallRequest(
                    id=_short_tool_id(),
                    name=str(function_payload.get("name") or ""),
                    arguments=arguments if isinstance(arguments, dict) else {},
                    extra_content=extra_content,
                    provider_specific_fields=provider_fields,
                    function_provider_specific_fields=function_provider_fields,
                ))

            return LLMResponse(
                content=content,
                tool_calls=parsed_tool_calls,
                finish_reason=finish_reason,
                usage=self._extract_usage(response_map),
                reasoning_content=reasoning_content if isinstance(reasoning_content, str) else None,
                error_detail=(
                    self._provider_error_detail(response_map, choice0)
                    if finish_reason == "error"
                    else None
                ),
            )

        if not provider_response.choices:
            return LLMResponse(content="Error: API returned empty choices.", finish_reason="error")

        first_choice = provider_response.choices[0]
        message = first_choice.message
        content = message.content
        finish_reason = first_choice.finish_reason

        raw_tool_calls: list[Any] = []
        for choice in provider_response.choices:
            message = choice.message
            if hasattr(message, "tool_calls") and message.tool_calls:
                raw_tool_calls.extend(message.tool_calls)
                if choice.finish_reason in ("tool_calls", "stop"):
                    finish_reason = choice.finish_reason
            if not content and message.content:
                content = message.content

        tool_calls = []
        for tool_call in raw_tool_calls:
            arguments = tool_call.function.arguments
            if isinstance(arguments, str):
                arguments = json_repair.loads(arguments)
            extra_content, provider_fields, function_provider_fields = (
                _extract_tc_extras(tool_call)
            )
            tool_calls.append(ToolCallRequest(
                id=_short_tool_id(),
                name=tool_call.function.name,
                arguments=arguments,
                extra_content=extra_content,
                provider_specific_fields=provider_fields,
                function_provider_specific_fields=function_provider_fields,
            ))

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason=finish_reason or "stop",
            usage=self._extract_usage(provider_response),
            reasoning_content=self._extract_provider_reasoning(message),
        )

    @classmethod
    def _parse_chunks(cls, chunks: list[Any]) -> LLMResponse:
        content_parts: list[str] = []
        tool_call_buffers: dict[int, dict[str, Any]] = {}
        finish_reason = "stop"
        usage: dict[str, int] = {}

        def _accum_tc(tool_call: Any, index_hint: int) -> None:
            """Accumulate one streaming tool-call delta into its buffer."""
            tool_call_index: int = (
                _get(tool_call, "index")
                if _get(tool_call, "index") is not None
                else index_hint
            )
            tool_call_buffer = tool_call_buffers.setdefault(tool_call_index, {
                "id": "", "name": "", "arguments": "",
                "extra_content": None,
                "provider_fields": None,
                "function_provider_fields": None,
            })
            tool_call_id = _get(tool_call, "id")
            if tool_call_id:
                tool_call_buffer["id"] = str(tool_call_id)
            function_object = _get(tool_call, "function")
            if function_object is not None:
                function_name = _get(function_object, "name")
                if function_name:
                    tool_call_buffer["name"] = str(function_name)
                function_arguments = _get(function_object, "arguments")
                if function_arguments:
                    tool_call_buffer["arguments"] += str(function_arguments)
            extra_content, provider_fields, function_provider_fields = (
                _extract_tc_extras(tool_call)
            )
            if extra_content:
                tool_call_buffer["extra_content"] = extra_content
            if provider_fields:
                tool_call_buffer["provider_fields"] = provider_fields
            if function_provider_fields:
                tool_call_buffer["function_provider_fields"] = function_provider_fields

        for chunk in chunks:
            if isinstance(chunk, str):
                content_parts.append(chunk)
                continue

            chunk_map = cls._maybe_mapping(chunk)
            if chunk_map is not None:
                choices = chunk_map.get("choices") or []
                if not choices:
                    usage = cls._extract_usage(chunk_map) or usage
                    text = cls._extract_text_content(
                        chunk_map.get("content") or chunk_map.get("output_text")
                    )
                    if text:
                        content_parts.append(text)
                    continue
                choice = cls._maybe_mapping(choices[0]) or {}
                if choice.get("finish_reason"):
                    finish_reason = str(choice["finish_reason"])
                delta = cls._maybe_mapping(choice.get("delta")) or {}
                text = cls._extract_text_content(delta.get("content"))
                if text:
                    content_parts.append(text)
                for index, tool_call in enumerate(delta.get("tool_calls") or []):
                    _accum_tc(tool_call, index)
                usage = cls._extract_usage(chunk_map) or usage
                continue

            if not chunk.choices:
                usage = cls._extract_usage(chunk) or usage
                continue
            choice = chunk.choices[0]
            if choice.finish_reason:
                finish_reason = choice.finish_reason
            delta = choice.delta
            if delta and delta.content:
                content_parts.append(delta.content)
            for tool_call in (delta.tool_calls or []) if delta else []:
                _accum_tc(tool_call, getattr(tool_call, "index", 0))

        return LLMResponse(
            content="".join(content_parts) or None,
            tool_calls=[
                ToolCallRequest(
                    id=tool_call_buffer["id"] or _short_tool_id(),
                    name=tool_call_buffer["name"],
                    arguments=(
                        json_repair.loads(tool_call_buffer["arguments"])
                        if tool_call_buffer["arguments"]
                        else {}
                    ),
                    extra_content=tool_call_buffer.get("extra_content"),
                    provider_specific_fields=tool_call_buffer.get("provider_fields"),
                    function_provider_specific_fields=tool_call_buffer.get(
                        "function_provider_fields"
                    ),
                )
                for tool_call_buffer in tool_call_buffers.values()
            ],
            finish_reason=finish_reason,
            usage=usage,
        )

    @staticmethod
    def _recover_dsml_tool_calls(response: LLMResponse) -> LLMResponse:
        """The one seam every response passes through, streamed or not.

        `finish_reason` is left alone: the runner branches on `has_tool_calls`,
        and rewriting `stop` to `tool_calls` would only change what the audit
        line says the provider answered.
        """
        content, calls = convert_dsml_tool_calls(response.content)
        if not calls:
            return response
        response.content = content
        response.tool_calls = [*response.tool_calls, *calls]
        response.dsml_converted = len(calls)
        return response

    def _persist_raw(self, request_payload: dict[str, Any], started: float, *, response: Any = None, chunks: Any = None, error: BaseException | None = None) -> None:
        """Logging only: never raises, never changes the call's result."""
        if self.raw_response_log is None:
            return
        try:
            import json as _json
            def dump(obj: Any) -> Any:
                for attr in ("model_dump", "to_dict"):
                    if hasattr(obj, attr):
                        try:
                            return getattr(obj, attr)()
                        except Exception:
                            pass
                return repr(obj)
            row: dict[str, Any] = {
                "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "seconds": round(time.monotonic() - started, 3),
                "task": self._current_task_key(), "model": request_payload.get("model"),
                "tool_choice": request_payload.get("tool_choice"), "extra_body": request_payload.get("extra_body"),
                "max_tokens": request_payload.get("max_tokens"),
            }
            if response is not None:
                row["response"] = dump(response)
            if chunks is not None:
                row["stream_chunks"] = [dump(c) for c in chunks]
            if error is not None:
                raw = getattr(error, "response", None)
                row["error"] = {"type": type(error).__name__, "message": str(error), "status_code": getattr(error, "status_code", None),
                                "body": getattr(error, "body", None), "raw_text": getattr(raw, "text", None) if raw is not None else None}
            self.raw_response_log.parent.mkdir(parents=True, exist_ok=True)
            with self.raw_response_log.open("a", encoding="utf-8") as f:
                f.write(_json.dumps(row, ensure_ascii=False, default=repr) + "\n")
        except Exception:
            pass

    def _current_task_key(self) -> str | None:
        identity = self.current_task_identity
        return identity.logical_task_key if identity is not None else None

    def _record_upstream_provider(self, upstream_provider: str | None) -> None:
        if upstream_provider:
            self._upstream_providers.append(upstream_provider)

    def reset_upstream_providers(self) -> None:
        """Start a fresh window; the PA loop calls this once per turn."""
        self._upstream_providers.clear()

    def upstream_providers(self) -> list[str]:
        """The upstreams that served this window's calls, in order."""
        return list(self._upstream_providers)

    @staticmethod
    def _log_timeout(model: str, started: float) -> None:
        """The deadline expiring is the one failure with no response to log from."""
        _log_openrouter_call(
            model, started, LLMResponse(content=None, finish_reason="timeout"), None
        )

    @staticmethod
    def _handle_error(e: Exception) -> LLMResponse:
        body = getattr(e, "doc", None) or getattr(getattr(e, "response", None), "text", None)
        msg = f"Error: {body.strip()[:500]}" if body and body.strip() else f"Error calling LLM: {e}"
        return LLMResponse(
            content=msg,
            finish_reason="error",
            error_detail={
                "exception_type": type(e).__name__,
                "message": str(e),
                "status_code": getattr(e, "status_code", None),
                "response_body": body[:2_000] if isinstance(body, str) else None,
            },
        )

    def _handle_request_error(self, e: Exception, request_payload: dict[str, Any]) -> LLMResponse:
        """`_handle_error`, counted against the upstream the failed request names."""
        response = self._handle_error(e)
        upstream_provider = _error_upstream_provider(e)
        if upstream_provider:
            response.served_by = ServedBy(request_payload["model"], upstream_provider)
        self._count_arrival_failure(response)
        return response

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> LLMResponse:
        request_payload = self._build_kwargs(
            messages, tools, model, max_tokens, temperature,
            reasoning_effort, tool_choice,
        )
        return await self._send_chat(request_payload, may_reset_avoidance=True)

    async def _send_chat(
        self, request_payload: dict[str, Any], *, may_reset_avoidance: bool,
    ) -> LLMResponse:
        started = time.monotonic()
        try:
            provider_response = await self._client.chat.completions.create(
                **request_payload
            )
            self._persist_raw(request_payload, started, response=provider_response)
            response = self._recover_dsml_tool_calls(self._parse(provider_response))
            upstream_provider = self._extract_upstream_provider(provider_response)
            self._record_upstream_provider(upstream_provider)
            if upstream_provider:
                response.served_by = ServedBy(request_payload["model"], upstream_provider)
            _log_openrouter_call(
                request_payload["model"],
                started,
                response,
                self._extract_cost(provider_response),
                self._current_task_key(),
                upstream_provider,
                _request_provider_ignore(request_payload),
            )
        except APITimeoutError as e:
            self._persist_raw(request_payload, started, error=e)
            self._log_timeout(request_payload["model"], started)
            return self._handle_error(e)
        except Exception as e:
            self._persist_raw(request_payload, started, error=e)
            if may_reset_avoidance and self._no_upstream_left(e):
                return await self._send_chat(
                    self._reset_avoidance(request_payload), may_reset_avoidance=False,
                )
            return self._handle_request_error(e, request_payload)
        # Outside the `try`: a state file that cannot be updated must stop the
        # run, not come back as a provider error.
        self._count_arrival_failure(response)
        return response

    async def chat_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        on_content_delta: Callable[[str], Awaitable[None]] | None = None,
    ) -> LLMResponse:
        request_payload = self._build_kwargs(
            messages, tools, model, max_tokens, temperature,
            reasoning_effort, tool_choice,
        )
        request_payload["stream"] = True
        request_payload["stream_options"] = {"include_usage": True}
        return await self._send_chat_stream(
            request_payload, on_content_delta, may_reset_avoidance=True,
        )

    async def _send_chat_stream(
        self,
        request_payload: dict[str, Any],
        on_content_delta: Callable[[str], Awaitable[None]] | None,
        *,
        may_reset_avoidance: bool,
    ) -> LLMResponse:
        started = time.monotonic()
        try:
            stream = await self._client.chat.completions.create(**request_payload)
            chunks: list[Any] = []
            async for chunk in stream:
                chunks.append(chunk)
                if on_content_delta and chunk.choices:
                    text = getattr(chunk.choices[0].delta, "content", None)
                    if text:
                        await on_content_delta(text)
            self._persist_raw(request_payload, started, chunks=chunks)
            response = self._recover_dsml_tool_calls(self._parse_chunks(chunks))
            upstream_provider = next(
                (
                    name
                    for name in (
                        self._extract_upstream_provider(c) for c in chunks
                    )
                    if name is not None
                ),
                None,
            )
            self._record_upstream_provider(upstream_provider)
            if upstream_provider:
                response.served_by = ServedBy(request_payload["model"], upstream_provider)
            _log_openrouter_call(
                request_payload["model"],
                started,
                response,
                next(
                    (
                        cost
                        for cost in (self._extract_cost(c) for c in reversed(chunks))
                        if cost is not None
                    ),
                    None,
                ),
                self._current_task_key(),
                upstream_provider,
                _request_provider_ignore(request_payload),
            )
        except APITimeoutError as e:
            self._persist_raw(request_payload, started, error=e)
            self._log_timeout(request_payload["model"], started)
            return self._handle_error(e)
        except Exception as e:
            self._persist_raw(request_payload, started, error=e)
            if may_reset_avoidance and self._no_upstream_left(e):
                return await self._send_chat_stream(
                    self._reset_avoidance(request_payload), on_content_delta,
                    may_reset_avoidance=False,
                )
            return self._handle_request_error(e, request_payload)
        # Outside the `try`: a state file that cannot be updated must stop the
        # run, not come back as a provider error.
        self._count_arrival_failure(response)
        return response

    def get_default_model(self) -> str:
        return self.default_model
