"""Strict HTTP client for the self-hosted MemOS service."""

from __future__ import annotations

from copy import deepcopy
import math
import time
from typing import Any, Protocol, TYPE_CHECKING
from urllib.parse import urlsplit

import httpx
from memory.backends.runtime_contracts import (
    MEMOS_MEMPA_PATCH_VERSION,
    MEMOS_SOURCE_SHA256,
    MEMOS_SUBMODULE_COMMIT,
    MEMOS_VERSION,
)

if TYPE_CHECKING:
    from memory.backends.memos_read_only import MemOSReadOnlyClient


# MemOS operations that read without changing state, so a transport timeout
# that may have been delivered can still be replayed once.
_READ_ONLY_OPERATIONS = frozenset({"health", "search", "get_memory"})
_RETRY_PAUSE_SECONDS = 2.0
# `/product/add` runs the extraction LLM and the graph writes server-side and
# cannot be replayed after a read timeout, so it waits far longer than the rest.
_ADD_TIMEOUT_SECONDS = 600.0


class MemOSClientError(RuntimeError):
    """Base error for MemOS transport and response-contract failures."""


class MemOSTransportError(MemOSClientError):
    """Raised when the MemOS service cannot complete an HTTP request."""


class MemOSResponseError(MemOSClientError):
    """Raised when MemOS returns an invalid or unsuccessful response."""


class MemOSClient(Protocol):
    """MemOS operations required by the benchmark adapter."""

    def health(self) -> dict[str, Any]: ...

    def add_memories(self, add_request: dict[str, Any]) -> list[dict[str, Any]]: ...

    def get_memories(self, get_request: dict[str, Any]) -> dict[str, Any]: ...

    def search_memories(self, search_request: dict[str, Any]) -> dict[str, Any]: ...

    def delete_memories(self, delete_request: dict[str, Any]) -> dict[str, Any]: ...

    def export_cube(self, cube_id: str) -> dict[str, Any]: ...

    def delete_cube(self, cube_id: str) -> dict[str, Any]: ...

    def import_cube_as(
        self, payload: dict[str, Any], target_cube_id: str
    ) -> dict[str, Any]: ...


class MemOSHttpClient:
    """Synchronous client for MemOS's `/product` REST API.

    `health()` returns the service's `runtime_contract` (source commit and
    digest, models, embedding, images) for the caller to record.
    `expected_runtime_contract` is accepted from callers that still pass the
    plan's contract; it is not compared with the service's.
    """

    def __init__(
        self,
        *,
        api_base: str,
        timeout_seconds: float = 180.0,
        expected_runtime_contract: dict[str, Any] | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        normalized_api_base = api_base.strip().rstrip("/")
        parsed = urlsplit(normalized_api_base)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError(
                "MemOS API base must be an absolute HTTP(S) URL: "
                f"value={api_base!r}"
            )
        if parsed.username is not None or parsed.password is not None:
            raise ValueError(
                "MemOS API base must not contain credentials: "
                f"value={normalized_api_base!r}"
            )
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError(
                "MemOS timeout must be finite and positive: "
                f"value={timeout_seconds!r}"
            )
        self._api_base = normalized_api_base
        self._timeout_seconds = timeout_seconds
        self._expected_runtime_contract = deepcopy(expected_runtime_contract)
        self._transport = transport

    @property
    def api_base(self) -> str:
        return self._api_base

    def read_only_for_cube(self, cube_id: str) -> MemOSReadOnlyClient:
        """Open one owner-managed query connection pool for a prepared cube.

        The caller keeps this client across both probe repeats and calls
        ``close`` only after all consumers have finished. Creating it does not
        restore, initialize, or otherwise write the cube.
        """
        from memory.backends.memos_read_only import MemOSReadOnlyClient

        return MemOSReadOnlyClient(
            cube_id=cube_id,
            api_base=self._api_base,
            timeout_seconds=self._timeout_seconds,
            expected_runtime_contract=self._expected_runtime_contract,
            transport=self._transport,
        )

    def health(self) -> dict[str, Any]:
        payload = self._request_json("GET", "/health", operation="health")
        if not isinstance(payload, dict):
            raise self._response_error(
                "health", f"response root must be an object: value={payload!r}"
            )
        status = payload.get("status")
        service = payload.get("service")
        version = payload.get("version")
        patch_version = payload.get("mempa_patch_version")
        runtime_contract = payload.get("runtime_contract")
        dependencies = payload.get("dependencies")
        if (
            status != "healthy"
            or service != "memos"
            or not isinstance(version, str)
            or patch_version != MEMOS_MEMPA_PATCH_VERSION
        ):
            raise self._response_error(
                "health",
                "invalid health response: "
                f"status={status!r} service={service!r} version={version!r} "
                f"expected_mempa_patch_version="
                f"{MEMOS_MEMPA_PATCH_VERSION!r} "
                f"mempa_patch_version={patch_version!r}",
            )
        if not isinstance(runtime_contract, dict):
            raise self._response_error(
                "health",
                "runtime_contract must be an object: "
                f"value_type={type(runtime_contract).__name__}",
            )
        if dependencies != {"qdrant": "healthy", "neo4j": "healthy"}:
            raise self._response_error(
                "health",
                "dependency health is invalid: "
                f"value={dependencies!r} expected={{'qdrant': 'healthy', "
                "'neo4j': 'healthy'}}",
            )
        return payload

    def add_memories(self, add_request: dict[str, Any]) -> list[dict[str, Any]]:
        requested_cube_ids = add_request.get("writable_cube_ids")
        if (
            not isinstance(requested_cube_ids, list)
            or not requested_cube_ids
            or any(
                not isinstance(cube_id, str) or not cube_id.strip()
                for cube_id in requested_cube_ids
            )
        ):
            raise ValueError(
                "MemOS add writable_cube_ids must be a non-empty list of "
                f"identifiers: value={requested_cube_ids!r}"
            )
        normalized_cube_ids = [cube_id.strip() for cube_id in requested_cube_ids]
        if len(set(normalized_cube_ids)) != len(normalized_cube_ids):
            raise ValueError(
                "MemOS add writable_cube_ids must be unique: "
                f"value={normalized_cube_ids!r}"
            )

        data = self._post_product("add", add_request)
        if not isinstance(data, list):
            raise self._response_error(
                "add", f"data must be a list: value_type={type(data).__name__}"
            )
        validated: list[dict[str, Any]] = []
        seen_memory_ids: set[str] = set()
        for index, item in enumerate(data):
            if not isinstance(item, dict):
                raise self._response_error(
                    "add",
                    f"data[{index}] must be an object: value={item!r}",
                )
            response_id = item.get("id")
            response_memory_id = item.get("memory_id")
            if (
                response_id is not None
                and response_memory_id is not None
                and response_id != response_memory_id
            ):
                raise self._response_error(
                    "add",
                    f"data[{index}] contains conflicting id fields: "
                    f"id={response_id!r} memory_id={response_memory_id!r}",
                )
            memory_id = item.get("memory_id") or item.get("id")
            if not isinstance(memory_id, str) or not memory_id.strip():
                raise self._response_error(
                    "add",
                    f"data[{index}] requires a non-empty memory_id/id: value={item!r}",
                )
            normalized_memory_id = memory_id.strip()
            if normalized_memory_id in seen_memory_ids:
                raise self._response_error(
                    "add",
                    "response contains a duplicate memory id: "
                    f"value={normalized_memory_id!r} index={index!r}",
                )
            seen_memory_ids.add(normalized_memory_id)

            memory_text = item.get("memory")
            if not isinstance(memory_text, str) or not memory_text.strip():
                raise self._response_error(
                    "add",
                    f"data[{index}] requires non-empty memory text: "
                    f"value={memory_text!r}",
                )
            cube_id = item.get("cube_id")
            if (
                not isinstance(cube_id, str)
                or cube_id.strip() not in normalized_cube_ids
            ):
                raise self._response_error(
                    "add",
                    f"data[{index}].cube_id={cube_id!r} does not match expected "
                    f"writable_cube_ids={normalized_cube_ids!r}",
                )
            validated.append(item)
        return validated

    def get_memories(self, get_request: dict[str, Any]) -> dict[str, Any]:
        requested_cube_id = get_request.get("mem_cube_id")
        if not isinstance(requested_cube_id, str) or not requested_cube_id.strip():
            raise ValueError(
                "MemOS get mem_cube_id must be a non-empty identifier: "
                f"value={requested_cube_id!r}"
            )
        expected_cube_ids = {requested_cube_id.strip()}
        data = self._post_product("get_memory", get_request)
        return self._validate_grouped_memory_data(
            "get_memory", data, expected_cube_ids=expected_cube_ids
        )

    def search_memories(self, search_request: dict[str, Any]) -> dict[str, Any]:
        requested_cube_ids = search_request.get("readable_cube_ids")
        if (
            not isinstance(requested_cube_ids, list)
            or not requested_cube_ids
            or any(
                not isinstance(cube_id, str) or not cube_id.strip()
                for cube_id in requested_cube_ids
            )
        ):
            raise ValueError(
                "MemOS search readable_cube_ids must be a non-empty list of "
                f"identifiers: value={requested_cube_ids!r}"
            )
        normalized_cube_ids = [cube_id.strip() for cube_id in requested_cube_ids]
        if len(set(normalized_cube_ids)) != len(normalized_cube_ids):
            raise ValueError(
                "MemOS search readable_cube_ids must be unique: "
                f"value={normalized_cube_ids!r}"
            )
        data = self._post_product("search", search_request)
        return self._validate_grouped_memory_data(
            "search", data, expected_cube_ids=set(normalized_cube_ids)
        )

    def delete_memories(self, delete_request: dict[str, Any]) -> dict[str, Any]:
        data = self._post_product("delete_memory", delete_request)
        if not isinstance(data, dict):
            raise self._response_error(
                "delete_memory",
                f"data must be an object: value_type={type(data).__name__}",
            )
        if data.get("status") != "success":
            raise self._response_error(
                "delete_memory",
                f"delete did not succeed: status={data.get('status')!r}",
            )
        deleted_count = data.get("deleted_count")
        if isinstance(deleted_count, bool) or not isinstance(deleted_count, int):
            raise self._response_error(
                "delete_memory",
                "deleted_count must be an integer: "
                f"value={deleted_count!r} value_type={type(deleted_count).__name__}",
            )
        if deleted_count < 0:
            raise self._response_error(
                "delete_memory",
                f"deleted_count must be non-negative: value={deleted_count!r}",
            )
        requested_ids = delete_request.get("memory_ids")
        if isinstance(requested_ids, list):
            expected_count = len(set(requested_ids))
            if deleted_count != expected_count:
                raise self._response_error(
                    "delete_memory",
                    "deleted_count does not match explicit memory_ids: "
                    f"expected_count={expected_count} deleted_count={deleted_count}",
                )
        return data

    # ── checkpoint support ────────────────────────────────────────
    #
    # A cut point captures one whole cube, and a probe branch is restored into
    # a cube of its own so nothing it writes reaches the mainline cube.

    def export_cube(self, cube_id: str) -> dict[str, Any]:
        """Export one whole cube; the payload is what a restore feeds back."""
        data = self._post_product(
            "export_cube", {"mem_cube_id": _validated_cube_id("export_cube", cube_id)}
        )
        if not isinstance(data, dict):
            raise self._response_error(
                "export_cube",
                f"data must be an object: value_type={type(data).__name__}",
            )
        return data

    def delete_cube(self, cube_id: str) -> dict[str, Any]:
        """Drop one whole cube, not the memories inside it."""
        data = self._post_product(
            "delete_cube", {"mem_cube_id": _validated_cube_id("delete_cube", cube_id)}
        )
        if not isinstance(data, dict):
            raise self._response_error(
                "delete_cube",
                f"data must be an object: value_type={type(data).__name__}",
            )
        if data.get("status") != "success":
            raise self._response_error(
                "delete_cube",
                f"delete did not succeed: status={data.get('status')!r}",
            )
        return data

    def import_cube_as(
        self, payload: dict[str, Any], target_cube_id: str
    ) -> dict[str, Any]:
        """Restore an exported cube; the service validates before replacing data.

        The Community deployment requires the original cube id so native IDs
        cannot collide with another cube in the shared database.
        """
        if not isinstance(payload, dict):
            raise TypeError(
                "MemOS import_cube payload must be a dict: "
                f"value_type={type(payload).__name__}"
            )
        data = self._post_product(
            "import_cube",
            {
                "mem_cube_id": _validated_cube_id("import_cube", target_cube_id),
                "cube": payload,
            },
        )
        if not isinstance(data, dict):
            raise self._response_error(
                "import_cube",
                f"data must be an object: value_type={type(data).__name__}",
            )
        # The service counts what it actually restored; a half-written cube must
        # never look like a success here.
        for field, expected in (
            ("node_count", len(payload.get("nodes") or [])),
            ("point_count", len(payload.get("points") or [])),
        ):
            reported = data.get(field)
            if reported != expected:
                raise self._response_error(
                    "import_cube",
                    f"restored {field} does not match the snapshot: "
                    f"restored={reported!r} expected={expected!r}",
                )
        return data

    def _post_product(self, operation: str, request: dict[str, Any]) -> Any:
        if not isinstance(request, dict):
            raise TypeError(
                f"MemOS {operation} request must be a dict: "
                f"value_type={type(request).__name__}"
            )
        payload = self._request_json(
            "POST",
            f"/product/{operation}",
            operation=operation,
            json_body=request,
        )
        if not isinstance(payload, dict):
            raise self._response_error(
                operation,
                f"response root must be an object: value_type={type(payload).__name__}",
            )
        code = payload.get("code")
        message = payload.get("message")
        if code != 200:
            raise self._response_error(
                operation,
                f"API code is not 200: code={code!r} message={message!r}",
            )
        if not isinstance(message, str) or not message.strip():
            raise self._response_error(
                operation,
                f"response requires a non-empty message: value={message!r}",
            )
        if "data" not in payload:
            raise self._response_error(operation, "response is missing data")
        return payload["data"]

    @staticmethod
    def _may_replay(operation: str, error: httpx.HTTPError) -> bool:
        """Whether one replay of `operation` after `error` cannot double-write.

        A connect or pool timeout never reached MemOS, so the request cannot
        have been started and any operation may be replayed. A read timeout
        means the request was delivered and MemOS may still be working on it,
        so only operations that cannot change state may be replayed: MemOS
        mints a fresh UUID per extracted memory
        (`memories/textual/item.py:305`) and `/product/add` carries no dedupe
        key, so replaying an add after a late server-side completion would
        write the same memories twice.
        """
        if isinstance(error, (httpx.ConnectTimeout, httpx.PoolTimeout)):
            return True
        return (
            isinstance(error, httpx.ReadTimeout)
            and operation in _READ_ONLY_OPERATIONS
        )

    def _request_json(
        self,
        method: str,
        path: str,
        *,
        operation: str,
        json_body: dict[str, Any] | None = None,
    ) -> Any:
        timeout = (
            _ADD_TIMEOUT_SECONDS
            if operation == "add"
            else self._timeout_seconds
        )
        for attempt in (1, 2):
            try:
                response = self._send_request(
                    method, path, json_body=json_body, timeout=timeout
                )
                break
            except httpx.HTTPError as exc:
                if attempt == 1 and self._may_replay(operation, exc):
                    print(
                        f"memos_transport_retry op={operation} attempt=2 "
                        f"reason={type(exc).__name__}: {exc}",
                        flush=True,
                    )
                    time.sleep(_RETRY_PAUSE_SECONDS)
                    continue
                raise MemOSTransportError(
                    "MemOS request failed: "
                    f"operation={operation!r} api_base={self._api_base!r} "
                    f"error={type(exc).__name__}: {exc}"
                ) from exc
        if not 200 <= response.status_code < 300:
            preview = response.text[:240]
            raise MemOSResponseError(
                "MemOS HTTP request failed: "
                f"operation={operation!r} api_base={self._api_base!r} "
                f"status_code={response.status_code!r} response={preview!r}"
            )
        try:
            return response.json()
        except ValueError as exc:
            preview = response.text[:240]
            raise MemOSResponseError(
                "MemOS response is not valid JSON: "
                f"operation={operation!r} status_code={response.status_code!r} "
                f"response={preview!r}"
            ) from exc

    def _send_request(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None,
        timeout: float,
    ) -> httpx.Response:
        with httpx.Client(
            base_url=self._api_base,
            timeout=timeout,
            transport=self._transport,
        ) as client:
            return client.request(method, path, json=json_body)

    def _validate_grouped_memory_data(
        self,
        operation: str,
        data: Any,
        *,
        expected_cube_ids: set[str],
    ) -> dict[str, Any]:
        if not isinstance(data, dict):
            raise self._response_error(
                operation,
                f"data must be an object: value_type={type(data).__name__}",
            )
        group_names = (
            "text_mem",
            "act_mem",
            "para_mem",
            "pref_mem",
            "tool_mem",
            "skill_mem",
        )
        for group_name in group_names:
            buckets = data.get(group_name, [])
            if not isinstance(buckets, list):
                raise self._response_error(
                    operation,
                    f"{group_name} must be a list: value={buckets!r}",
                )
            for bucket_index, bucket in enumerate(buckets):
                if not isinstance(bucket, dict):
                    raise self._response_error(
                        operation,
                        f"{group_name}[{bucket_index}] must be an object: "
                        f"value={bucket!r}",
                    )
                cube_id = bucket.get("cube_id")
                memories = bucket.get("memories")
                if not isinstance(cube_id, str) or not cube_id.strip():
                    raise self._response_error(
                        operation,
                        f"{group_name}[{bucket_index}] requires cube_id: "
                        f"value={cube_id!r}",
                    )
                if cube_id.strip() not in expected_cube_ids:
                    raise self._response_error(
                        operation,
                        f"{group_name}[{bucket_index}].cube_id={cube_id!r} "
                        "does not match requested cube ids: "
                        f"expected={sorted(expected_cube_ids)!r}",
                    )
                if not isinstance(memories, list):
                    raise self._response_error(
                        operation,
                        f"{group_name}[{bucket_index}].memories must be a list: "
                        f"value={memories!r}",
                    )
                for memory_index, memory in enumerate(memories):
                    if not isinstance(memory, dict):
                        raise self._response_error(
                            operation,
                            f"{group_name}[{bucket_index}].memories[{memory_index}] "
                            f"must be an object: value={memory!r}",
                        )
                    memory_id = memory.get("id") or memory.get("memory_id")
                    memory_text = memory.get("memory")
                    if not isinstance(memory_id, str) or not memory_id.strip():
                        raise self._response_error(
                            operation,
                            f"{group_name}[{bucket_index}].memories[{memory_index}] "
                            f"requires a non-empty id: value={memory_id!r}",
                        )
                    if not isinstance(memory_text, str) or not memory_text.strip():
                        raise self._response_error(
                            operation,
                            f"{group_name}[{bucket_index}].memories[{memory_index}] "
                            f"requires non-empty memory text: value={memory_text!r}",
                        )
                    if operation == "search" and group_name == "text_mem":
                        metadata = memory.get("metadata")
                        score = (
                            metadata.get("relativity")
                            if isinstance(metadata, dict)
                            else None
                        )
                        if (
                            isinstance(score, bool)
                            or not isinstance(score, (int, float))
                            or not math.isfinite(score)
                            or not 0 <= score <= 1
                        ):
                            raise self._response_error(
                                operation,
                                f"{group_name}[{bucket_index}].memories"
                                f"[{memory_index}].metadata.relativity must be "
                                "a finite number between 0 and 1: "
                                f"value={score!r}",
                            )
        pref_note = data.get("pref_note")
        if pref_note is not None and not isinstance(pref_note, str):
            raise self._response_error(
                operation,
                f"pref_note must be a string when present: value={pref_note!r}",
            )
        return data

    def _response_error(self, operation: str, detail: str) -> MemOSResponseError:
        return MemOSResponseError(
            "Invalid MemOS response: "
            f"operation={operation!r} api_base={self._api_base!r} {detail}"
        )


def _validated_cube_id(operation: str, cube_id: str) -> str:
    if not isinstance(cube_id, str) or not cube_id.strip():
        raise ValueError(
            f"MemOS {operation} requires a non-empty cube id: value={cube_id!r}"
        )
    return cube_id.strip()
