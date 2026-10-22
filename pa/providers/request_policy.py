"""Typed request policies for provider/model pairs."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


DEFAULT_MODEL_POLICY_CATALOG = (
    Path(__file__).resolve().parents[2] / "configs/model_request_policies.yaml"
)
LLMRole = Literal["pa", "consolidation", "memory_extraction"]


class ModelRequestPolicyError(ValueError):
    """Raised when a model request policy cannot be resolved safely."""


class ModelRequestPolicy(BaseModel):
    """Resolved request contract for one provider/model pair."""

    model_config = ConfigDict(extra="forbid")

    provider_name: str
    model: str
    temperature: Literal["send", "omit"] = "send"
    completion_tokens_parameter: Literal[
        "max_tokens",
        "max_completion_tokens",
    ] = "max_tokens"
    reasoning_effort: Literal["send", "omit"] = "send"
    supports_tools: bool = True
    supports_forced_tool_choice: bool = True
    validated_roles: list[LLMRole] = Field(default_factory=list)
    policy_source: Literal["model", "provider_default"] = "model"

    def snapshot(self) -> dict[str, object]:
        return self.model_dump(mode="json")

    def require_role(self, role: LLMRole) -> None:
        if role not in self.validated_roles:
            raise ModelRequestPolicyError(
                "Model request policy is not validated for role: "
                f"provider_name={self.provider_name!r} model={self.model!r} "
                f"role={role!r} validated_roles={self.validated_roles!r}"
            )

    def require_exact_role(self, role: LLMRole) -> None:
        self.require_role(role)
        if self.policy_source != "model":
            raise ModelRequestPolicyError(
                "Benchmark role requires an exact model request policy: "
                f"provider_name={self.provider_name!r} model={self.model!r} "
                f"role={role!r} policy_source={self.policy_source!r}"
            )


class ProviderDefaultPolicy(BaseModel):
    """Default request contract for a provider with uniform model behavior."""

    model_config = ConfigDict(extra="forbid")

    provider_name: str
    temperature: Literal["send", "omit"] = "send"
    completion_tokens_parameter: Literal[
        "max_tokens",
        "max_completion_tokens",
    ] = "max_tokens"
    reasoning_effort: Literal["send", "omit"] = "send"
    supports_tools: bool = True
    supports_forced_tool_choice: bool = True
    validated_roles: list[LLMRole] = Field(default_factory=list)

    def resolve(self, model: str) -> ModelRequestPolicy:
        return ModelRequestPolicy(
            **self.model_dump(),
            model=model,
            policy_source="provider_default",
        )


class ProviderRouting(BaseModel):
    """Upstream routing for one provider/model pair.

    Sent as the request body's `provider` field (OpenRouter's provider
    preferences). Kept out of `ModelRequestPolicy`, so it never changes a
    policy snapshot or its sha256.
    """

    model_config = ConfigDict(extra="forbid")

    provider_name: str
    model: str
    order: list[str]
    allow_fallbacks: bool = False
    #: Logging only: when set, every PA call's raw upstream response is
    #: appended under this directory, relative to the `workspace` directory
    #: that holds the run (outside the run, so a restore never rolls it back).
    raw_response_log_dir: str | None = None

    def request_body(self) -> dict[str, object]:
        return {
            "provider": {"order": list(self.order), "allow_fallbacks": self.allow_fallbacks}
        }


class ModelPolicyCatalog(BaseModel):
    """Schema for the model request-policy source of truth."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1]
    provider_defaults: list[ProviderDefaultPolicy] = Field(default_factory=list)
    models: list[ModelRequestPolicy] = Field(default_factory=list)
    provider_routing: list[ProviderRouting] = Field(default_factory=list)

    @model_validator(mode="after")
    def _unique_policy_keys(self) -> ModelPolicyCatalog:
        default_names = [item.provider_name for item in self.provider_defaults]
        duplicate_defaults = sorted(
            name for name in set(default_names) if default_names.count(name) > 1
        )
        if duplicate_defaults:
            raise ValueError(
                f"duplicate provider default policies: value={duplicate_defaults!r}"
            )

        model_keys = [(item.provider_name, item.model) for item in self.models]
        duplicate_models = sorted(
            key for key in set(model_keys) if model_keys.count(key) > 1
        )
        if duplicate_models:
            raise ValueError(
                f"duplicate provider/model policies: value={duplicate_models!r}"
            )

        routing_keys = [(item.provider_name, item.model) for item in self.provider_routing]
        duplicate_routing = sorted(
            key for key in set(routing_keys) if routing_keys.count(key) > 1
        )
        if duplicate_routing:
            raise ValueError(
                f"duplicate provider/model routing: value={duplicate_routing!r}"
            )
        return self

    def resolve(self, provider_name: str, model: str) -> ModelRequestPolicy:
        for policy in self.models:
            if policy.provider_name == provider_name and policy.model == model:
                return policy
        for provider_default in self.provider_defaults:
            if provider_default.provider_name == provider_name:
                return provider_default.resolve(model)
        raise ModelRequestPolicyError(
            "No model request policy registered: "
            f"provider_name={provider_name!r} model={model!r}"
        )


def load_model_policy_catalog(
    path: Path | str = DEFAULT_MODEL_POLICY_CATALOG,
) -> ModelPolicyCatalog:
    source_path = Path(path)
    try:
        text = source_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ModelRequestPolicyError(
            f"Failed to read model policy catalog: path={source_path} "
            f"error={type(exc).__name__}: {exc}"
        ) from exc
    try:
        raw_catalog = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ModelRequestPolicyError(
            f"Invalid model policy YAML: path={source_path} "
            f"error={type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(raw_catalog, dict):
        raise ModelRequestPolicyError(
            f"Model policy catalog root must be a mapping: path={source_path} "
            f"root_type={type(raw_catalog).__name__} value={raw_catalog!r}"
        )
    try:
        return ModelPolicyCatalog.model_validate(raw_catalog)
    except ValidationError as exc:
        raise ModelRequestPolicyError(
            f"Invalid model policy catalog: path={source_path} error={exc}"
        ) from exc


def resolve_model_request_policy(
    provider_name: str,
    model: str,
    *,
    catalog_path: Path | str = DEFAULT_MODEL_POLICY_CATALOG,
) -> ModelRequestPolicy:
    return load_model_policy_catalog(catalog_path).resolve(provider_name, model)


def resolve_provider_routing(
    provider_name: str,
    model: str,
    *,
    catalog_path: Path | str = DEFAULT_MODEL_POLICY_CATALOG,
) -> ProviderRouting | None:
    """The configured upstream routing for this pair, or None for the default."""
    for routing in load_model_policy_catalog(catalog_path).provider_routing:
        if routing.provider_name == provider_name and routing.model == model:
            return routing
    return None


def validate_policy_snapshot(
    snapshot: dict[str, object],
    *,
    field_name: str,
) -> ModelRequestPolicy:
    """Parse a frozen policy snapshot."""
    try:
        return ModelRequestPolicy.model_validate(snapshot)
    except ValidationError as exc:
        raise ModelRequestPolicyError(
            f"Invalid frozen policy snapshot: field={field_name!r} "
            f"value={snapshot!r} error={exc}"
        ) from exc
