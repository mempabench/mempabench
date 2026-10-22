"""Cube-scoped query transport held by the owner of a shared probe setting."""

from __future__ import annotations

from copy import deepcopy
import math
from threading import Condition
from typing import Any

import httpx

from memory.backends.memos_client import (
    MemOSClientError,
    MemOSHttpClient,
    _validated_cube_id,
)


class MemOSUnresolvedRequestError(MemOSClientError):
    """A delivered request may still be running; its cube must be retained."""


class MemOSReadOnlyClient(MemOSHttpClient):
    """A shared connection pool with an allowlist at the request boundary.

    Each question gets its own adapter around this client. The client stores
    no query results, session state, or usage reporter. ``drain`` stops new
    requests and waits for synchronous HTTP workers, including workers whose
    asyncio callers were cancelled. A transport failure that may have reached
    the service prevents a successful drain, even if the usual retry succeeds.
    """

    read_only = True

    def __init__(self, *, cube_id: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._cube_id = _validated_cube_id("read_only_for_cube", cube_id)
        self._condition = Condition()
        self._accepting = True
        self._closed = False
        self._active_requests = 0
        self._unresolved_remote_requests = 0
        self._http = httpx.Client(
            base_url=self._api_base,
            timeout=self._timeout_seconds,
            transport=self._transport,
        )

    @property
    def cube_id(self) -> str:
        return self._cube_id

    @property
    def active_requests(self) -> int:
        with self._condition:
            return self._active_requests

    @property
    def unresolved_remote_requests(self) -> int:
        with self._condition:
            return self._unresolved_remote_requests

    def read_only_for_cube(self, cube_id: str) -> MemOSReadOnlyClient:
        raise PermissionError("Only the MemOS setting owner may create query clients")

    def ensure_query_ready(self) -> None:
        """Check admission locally, without opening or querying any resource."""
        with self._condition:
            if self._unresolved_remote_requests:
                raise MemOSUnresolvedRequestError(
                    "MemOS requests may still be running on the service; retain the shared cube: "
                    f"count={self._unresolved_remote_requests}"
                )
            if not self._accepting or self._closed:
                raise MemOSClientError("MemOS shared query client is draining or closed")

    def _check_request(
        self, method: str, path: str, body: dict[str, Any] | None
    ) -> None:
        if method == "GET" and path == "/health" and body is None:
            return
        if method == "POST" and isinstance(body, dict):
            if path == "/product/search" and body.get("readable_cube_ids") == [self._cube_id]:
                return
            if path == "/product/get_memory" and body.get("mem_cube_id") == self._cube_id:
                return
        raise PermissionError(
            "MemOS shared query client rejects write, management, or foreign-cube "
            f"request: method={method!r} path={path!r} cube_id={self._cube_id!r}"
        )

    def _request_json(
        self,
        method: str,
        path: str,
        *,
        operation: str,
        json_body: dict[str, Any] | None = None,
    ) -> Any:
        # Keep validation and serialization bound to the same request snapshot.
        body = deepcopy(json_body)
        self._check_request(method, path, body)
        expected = "health" if path == "/health" else path.removeprefix("/product/")
        if operation != expected:
            raise PermissionError("MemOS query operation does not match its allowed route")
        with self._condition:
            self.ensure_query_ready()
            self._active_requests += 1
        try:
            return super()._request_json(method, path, operation=operation, json_body=body)
        finally:
            with self._condition:
                self._active_requests -= 1
                self._condition.notify_all()

    def _send_request(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None,
        timeout: float,
    ) -> httpx.Response:
        self._check_request(method, path, json_body)
        try:
            return self._http.request(method, path, json=json_body, timeout=timeout)
        except httpx.HTTPError as exc:
            if not isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout)):
                with self._condition:
                    self._unresolved_remote_requests += 1
            raise

    def drain(self, timeout_seconds: float | None = None) -> None:
        """Quiesce permanently, or fail without authorizing cube deletion."""
        if timeout_seconds is not None and (
            not math.isfinite(timeout_seconds) or timeout_seconds < 0
        ):
            raise ValueError("drain timeout_seconds must be finite and non-negative")
        with self._condition:
            self._accepting = False
            if not self._condition.wait_for(
                lambda: self._active_requests == 0, timeout=timeout_seconds
            ):
                raise TimeoutError("MemOS HTTP workers have not drained; retain the shared cube")
            if self._unresolved_remote_requests:
                raise MemOSUnresolvedRequestError(
                    "MemOS requests may still be running on the service; retain the shared cube: "
                    f"count={self._unresolved_remote_requests}"
                )

    def close(self, timeout_seconds: float | None = None) -> None:
        """Drain and close the pool; unresolved remote work still raises."""
        try:
            self.drain(timeout_seconds)
        except MemOSUnresolvedRequestError:
            # Local workers are done, so connections can close, but the owner
            # still receives the error and must not reclaim the remote cube.
            self._close_pool()
            raise
        self._close_pool()

    def _close_pool(self) -> None:
        with self._condition:
            if not self._closed:
                self._http.close()
                self._closed = True
