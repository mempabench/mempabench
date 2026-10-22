"""Anthropic provider — direct SDK integration for Claude models."""

from __future__ import annotations

import re
import secrets
import string
from collections.abc import Awaitable, Callable
from typing import Any

import json_repair
from loguru import logger

from pa.providers.base import LLMProvider, LLMResponse, ToolCallRequest

_ALNUM = string.ascii_letters + string.digits


def _gen_tool_id() -> str:
    return "toolu_" + "".join(secrets.choice(_ALNUM) for _ in range(22))


class AnthropicProvider(LLMProvider):
    """LLM provider using the native Anthropic SDK for Claude models.

    Handles message format conversion (OpenAI → Anthropic Messages API),
    prompt caching, extended thinking, tool calls, and streaming.
    """

    def __init__(
        self,
        api_key: str | None = None,
        api_base: str | None = None,
        default_model: str = "claude-sonnet-4-20250514",
        extra_headers: dict[str, str] | None = None,
    ):
        super().__init__(api_key, api_base)
        self.default_model = default_model
        self.extra_headers = extra_headers or {}

        from anthropic import AsyncAnthropic

        client_options: dict[str, Any] = {}
        if api_key:
            client_options["api_key"] = api_key
        if api_base:
            client_options["base_url"] = api_base
        if extra_headers:
            client_options["default_headers"] = extra_headers
        self._client = AsyncAnthropic(**client_options)

    @staticmethod
    def _strip_prefix(model: str) -> str:
        if model.startswith("anthropic/"):
            return model[len("anthropic/"):]
        return model

    # ------------------------------------------------------------------
    # Message conversion: OpenAI chat format → Anthropic Messages API
    # ------------------------------------------------------------------

    def _convert_messages(
        self, messages: list[dict[str, Any]],
    ) -> tuple[str | list[dict[str, Any]], list[dict[str, Any]]]:
        """Return ``(system, anthropic_messages)``."""
        system: str | list[dict[str, Any]] = ""
        raw: list[dict[str, Any]] = []

        for message in messages:
            role = message.get("role", "")
            content = message.get("content")

            if role == "system":
                system = content if isinstance(content, (str, list)) else str(content or "")
                continue

            if role == "tool":
                block = self._tool_result_block(message)
                if raw and raw[-1]["role"] == "user":
                    prev_c = raw[-1]["content"]
                    if isinstance(prev_c, list):
                        prev_c.append(block)
                    else:
                        raw[-1]["content"] = [
                            {"type": "text", "text": prev_c or ""}, block,
                        ]
                else:
                    raw.append({"role": "user", "content": [block]})
                continue

            if role == "assistant":
                raw.append({
                    "role": "assistant",
                    "content": self._assistant_blocks(message),
                })
                continue

            if role == "user":
                raw.append({
                    "role": "user",
                    "content": self._convert_user_content(content),
                })
                continue

        return system, self._merge_consecutive(raw)

    @staticmethod
    def _tool_result_block(message: dict[str, Any]) -> dict[str, Any]:
        content = message.get("content")
        block: dict[str, Any] = {
            "type": "tool_result",
            "tool_use_id": message.get("tool_call_id", ""),
        }
        if isinstance(content, (str, list)):
            block["content"] = content
        else:
            block["content"] = str(content) if content else ""
        return block

    @staticmethod
    def _assistant_blocks(message: dict[str, Any]) -> list[dict[str, Any]]:
        blocks: list[dict[str, Any]] = []
        content = message.get("content")

        for thinking_block in message.get("thinking_blocks") or []:
            if (
                isinstance(thinking_block, dict)
                and thinking_block.get("type") == "thinking"
            ):
                blocks.append({
                    "type": "thinking",
                    "thinking": thinking_block.get("thinking", ""),
                    "signature": thinking_block.get("signature", ""),
                })

        if isinstance(content, str) and content:
            blocks.append({"type": "text", "text": content})
        elif isinstance(content, list):
            for item in content:
                blocks.append(item if isinstance(item, dict) else {"type": "text", "text": str(item)})

        for tool_call in message.get("tool_calls") or []:
            if not isinstance(tool_call, dict):
                continue
            function_payload = tool_call.get("function", {})
            arguments = function_payload.get("arguments", "{}")
            if isinstance(arguments, str):
                arguments = json_repair.loads(arguments)
            blocks.append({
                "type": "tool_use",
                "id": tool_call.get("id") or _gen_tool_id(),
                "name": function_payload.get("name", ""),
                "input": arguments,
            })

        return blocks or [{"type": "text", "text": ""}]

    def _convert_user_content(self, content: Any) -> Any:
        """Convert user message content, translating image_url blocks."""
        if isinstance(content, str) or content is None:
            return content or "(empty)"
        if not isinstance(content, list):
            return str(content)

        result: list[dict[str, Any]] = []
        for item in content:
            if not isinstance(item, dict):
                result.append({"type": "text", "text": str(item)})
                continue
            if item.get("type") == "image_url":
                converted = self._convert_image_block(item)
                if converted:
                    result.append(converted)
                continue
            result.append(item)
        return result or "(empty)"

    @staticmethod
    def _convert_image_block(block: dict[str, Any]) -> dict[str, Any] | None:
        """Convert OpenAI image_url block to Anthropic image block."""
        url = (block.get("image_url") or {}).get("url", "")
        if not url:
            return None
        data_url_match = re.match(r"data:(image/\w+);base64,(.+)", url, re.DOTALL)
        if data_url_match:
            return {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": data_url_match.group(1),
                    "data": data_url_match.group(2),
                },
            }
        return {
            "type": "image",
            "source": {"type": "url", "url": url},
        }

    @staticmethod
    def _merge_consecutive(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Anthropic requires alternating user/assistant roles."""
        merged: list[dict[str, Any]] = []
        for message in messages:
            if merged and merged[-1]["role"] == message["role"]:
                previous_content = merged[-1]["content"]
                current_content = message["content"]
                if isinstance(previous_content, str):
                    previous_content = [{"type": "text", "text": previous_content}]
                if isinstance(current_content, str):
                    current_content = [{"type": "text", "text": current_content}]
                if isinstance(current_content, list):
                    previous_content.extend(current_content)
                merged[-1]["content"] = previous_content
            else:
                merged.append(message)
        return merged

    # ------------------------------------------------------------------
    # Tool definition conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _convert_tools(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
        if not tools:
            return None
        result = []
        for tool in tools:
            function_payload = tool.get("function", tool)
            entry: dict[str, Any] = {
                "name": function_payload.get("name", ""),
                "input_schema": function_payload.get(
                    "parameters", {"type": "object", "properties": {}}
                ),
            }
            description = function_payload.get("description")
            if description:
                entry["description"] = description
            if "cache_control" in tool:
                entry["cache_control"] = tool["cache_control"]
            result.append(entry)
        return result

    @staticmethod
    def _convert_tool_choice(
        tool_choice: str | dict[str, Any] | None,
        thinking_enabled: bool = False,
    ) -> dict[str, Any] | None:
        if thinking_enabled:
            return {"type": "auto"}
        if tool_choice is None or tool_choice == "auto":
            return {"type": "auto"}
        if tool_choice == "required":
            return {"type": "any"}
        if tool_choice == "none":
            return None
        if isinstance(tool_choice, dict):
            name = tool_choice.get("function", {}).get("name")
            if name:
                return {"type": "tool", "name": name}
        return {"type": "auto"}

    # ------------------------------------------------------------------
    # Prompt caching
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_cache_control(
        system: str | list[dict[str, Any]],
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
    ) -> tuple[str | list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]] | None]:
        marker = {"type": "ephemeral"}

        if isinstance(system, str) and system:
            system = [{"type": "text", "text": system, "cache_control": marker}]
        elif isinstance(system, list) and system:
            system = list(system)
            system[-1] = {**system[-1], "cache_control": marker}

        new_msgs = list(messages)
        if len(new_msgs) >= 3:
            cached_message = new_msgs[-2]
            cached_content = cached_message.get("content")
            if isinstance(cached_content, str):
                new_msgs[-2] = {
                    **cached_message,
                    "content": [{
                        "type": "text",
                        "text": cached_content,
                        "cache_control": marker,
                    }],
                }
            elif isinstance(cached_content, list) and cached_content:
                cached_content_blocks = list(cached_content)
                cached_content_blocks[-1] = {
                    **cached_content_blocks[-1],
                    "cache_control": marker,
                }
                new_msgs[-2] = {
                    **cached_message,
                    "content": cached_content_blocks,
                }

        new_tools = tools
        if tools:
            new_tools = list(tools)
            new_tools[-1] = {**new_tools[-1], "cache_control": marker}

        return system, new_msgs, new_tools

    # ------------------------------------------------------------------
    # Build API kwargs
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
        supports_caching: bool = True,
    ) -> dict[str, Any]:
        model_name = self._strip_prefix(model or self.default_model)
        system, anthropic_msgs = self._convert_messages(self._sanitize_empty_content(messages))
        anthropic_tools = self._convert_tools(tools)

        if supports_caching:
            system, anthropic_msgs, anthropic_tools = self._apply_cache_control(
                system, anthropic_msgs, anthropic_tools,
            )

        max_tokens = max(1, max_tokens)
        thinking_enabled = bool(reasoning_effort)

        request_payload: dict[str, Any] = {
            "model": model_name,
            "messages": anthropic_msgs,
            "max_tokens": max_tokens,
        }

        if system:
            request_payload["system"] = system

        if thinking_enabled:
            budget_map = {"low": 1024, "medium": 4096, "high": max(8192, max_tokens)}
            budget = budget_map.get(reasoning_effort.lower(), 4096)  # type: ignore[union-attr]
            request_payload["thinking"] = {"type": "enabled", "budget_tokens": budget}
            request_payload["max_tokens"] = max(max_tokens, budget + 4096)
            request_payload["temperature"] = 1.0
        else:
            request_payload["temperature"] = temperature

        if anthropic_tools:
            request_payload["tools"] = anthropic_tools
            converted_tool_choice = self._convert_tool_choice(
                tool_choice, thinking_enabled
            )
            if converted_tool_choice:
                request_payload["tool_choice"] = converted_tool_choice

        if self.extra_headers:
            request_payload["extra_headers"] = self.extra_headers

        return request_payload

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_response(provider_response: Any) -> LLMResponse:
        content_parts: list[str] = []
        tool_calls: list[ToolCallRequest] = []
        thinking_blocks: list[dict[str, Any]] = []

        for block in provider_response.content:
            if block.type == "text":
                content_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(ToolCallRequest(
                    id=block.id,
                    name=block.name,
                    arguments=block.input if isinstance(block.input, dict) else {},
                ))
            elif block.type == "thinking":
                thinking_blocks.append({
                    "type": "thinking",
                    "thinking": block.thinking,
                    "signature": getattr(block, "signature", ""),
                })

        stop_map = {"tool_use": "tool_calls", "end_turn": "stop", "max_tokens": "length"}
        finish_reason = stop_map.get(
            provider_response.stop_reason or "",
            provider_response.stop_reason or "stop",
        )

        usage: dict[str, int] = {}
        if provider_response.usage:
            usage = {
                "prompt_tokens": provider_response.usage.input_tokens,
                "completion_tokens": provider_response.usage.output_tokens,
                "total_tokens": (
                    provider_response.usage.input_tokens
                    + provider_response.usage.output_tokens
                ),
            }
            for attr in ("cache_creation_input_tokens", "cache_read_input_tokens"):
                token_count = getattr(provider_response.usage, attr, 0)
                if token_count:
                    usage[attr] = token_count

        return LLMResponse(
            content="".join(content_parts) or None,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage=usage,
            thinking_blocks=thinking_blocks or None,
        )

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
        try:
            provider_response = await self._client.messages.create(**request_payload)
            return self._parse_response(provider_response)
        except Exception as e:
            return LLMResponse(content=f"Error calling LLM: {e}", finish_reason="error")

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
        try:
            async with self._client.messages.stream(**request_payload) as stream:
                if on_content_delta:
                    async for text in stream.text_stream:
                        await on_content_delta(text)
                provider_response = await stream.get_final_message()
            return self._parse_response(provider_response)
        except Exception as e:
            return LLMResponse(content=f"Error calling LLM: {e}", finish_reason="error")

    def get_default_model(self) -> str:
        return self.default_model
