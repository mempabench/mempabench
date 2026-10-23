"""OpenAI-compatible provider for all non-Anthropic LLM APIs."""

from __future__ import annotations

import hashlib
import os
import secrets
import string
import sys
import time
import uuid
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

import httpx
import json_repair
from openai import APITimeoutError, AsyncOpenAI

from simulator.providers.base import LLMProvider, LLMResponse, ToolCallRequest

if TYPE_CHECKING:
    from simulator.providers.registry import ProviderSpec

#: Wall-clock ceiling on one provider request. acc_039 sat nine
#: minutes on its first OpenRouter call with nothing on stderr: the SDK default
#: is a 600 s timeout with two internal retries, so a stalled connection could
#: hold a run for half an hour before anything surfaced. 180 s is the same
#: number RunPlan uses for `api_timeout_seconds`. `MEMPA_LLM_REQUEST_TIMEOUT_S`
#: overrides the total; the connect leg stays at 20 s.
#: Duplicated in pa/providers/openai_compat_provider.py on purpose: pa/ and
#: simulator/ never share modules.
_DEFAULT_REQUEST_TIMEOUT_S = 180.0
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
_ALNUM = string.ascii_letters + string.digits

_STANDARD_TC_KEYS = frozenset({"id", "type", "index", "function"})
_STANDARD_FN_KEYS = frozenset({"name", "arguments"})
_DEFAULT_OPENROUTER_HEADERS = {
    "HTTP-Referer": "https://github.com/HKUDS/nanobot",
    "X-OpenRouter-Title": "MemPABench-Simulator",
    "X-OpenRouter-Categories": "research,benchmark",
}


def _short_tool_id() -> str:
    return "".join(secrets.choice(_ALNUM) for _ in range(9))


def _get(obj: Any, key: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _coerce_dict(value: Any) -> dict[str, Any] | None:
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
    if spec and spec.name == "openrouter":
        return True
    return bool(api_base and "openrouter" in api_base.lower())



def _log_openrouter_call(model: str, started: float, response: LLMResponse, cost: Any) -> None:
    """One unbuffered stderr line per provider call, for live run monitoring.

    Duplicated in pa/providers/openai_compat_provider.py on purpose: pa/
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
    # One run on gemini-3.8-flash cut 7/13 self-report calls at completion_tokens=4092
    # and the log could not say `length`.
    line += f" finish_reason={response.finish_reason}"
    print(line, file=sys.stderr, flush=True)


class OpenAICompatProvider(LLMProvider):
    """Unified provider for all OpenAI-compatible APIs."""

    def __init__(
        self,
        api_key: str | None = None,
        api_base: str | None = None,
        default_model: str = "gpt-4o",
        extra_headers: dict[str, str] | None = None,
        spec: ProviderSpec | None = None,
    ):
        super().__init__(api_key, api_base)
        self.default_model = default_model
        self.extra_headers = extra_headers or {}
        self._spec = spec

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

    def _setup_env(self, api_key: str, api_base: str | None) -> None:
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
        if not isinstance(tool_call_id, str):
            return tool_call_id
        if len(tool_call_id) == 9 and tool_call_id.isalnum():
            return tool_call_id
        return hashlib.sha1(tool_call_id.encode()).hexdigest()[:9]

    def _sanitize_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
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

        if spec and spec.supports_prompt_caching:
            messages, tools = self._apply_cache_control(messages, tools)

        if spec and spec.strip_model_prefix:
            model_name = model_name.split("/")[-1]

        request_payload: dict[str, Any] = {
            "model": model_name,
            "messages": self._sanitize_messages(self._sanitize_empty_content(messages)),
            "temperature": temperature,
        }

        request_payload["max_tokens"] = max(1, max_tokens)

        if reasoning_effort:
            request_payload["reasoning_effort"] = reasoning_effort

        if tools:
            request_payload["tools"] = tools
            request_payload["tool_choice"] = tool_choice or "auto"

        return request_payload

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
            return {
                "prompt_tokens": int(usage_map.get("prompt_tokens") or 0),
                "completion_tokens": int(usage_map.get("completion_tokens") or 0),
                "total_tokens": int(usage_map.get("total_tokens") or 0),
            }

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
            reasoning_content = msg0.get("reasoning_content")
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
                    reasoning_content = message_map.get("reasoning_content")

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
            reasoning_content=getattr(message, "reasoning_content", None) or None,
        )

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
        started = time.monotonic()
        try:
            provider_response = await self._client.chat.completions.create(
                **request_payload
            )
            response = self._parse(provider_response)
            _log_openrouter_call(
                request_payload["model"],
                started,
                response,
                self._extract_cost(provider_response),
            )
            return response
        except APITimeoutError as e:
            self._log_timeout(request_payload["model"], started)
            return self._handle_error(e)
        except Exception as e:
            return self._handle_error(e)

    def get_default_model(self) -> str:
        return self.default_model
