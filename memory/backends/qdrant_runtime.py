"""Shared Qdrant transport and restore coordination."""

from __future__ import annotations

from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
from typing import Any, Iterator


QDRANT_OPERATION_TIMEOUT_SECONDS = 120
QDRANT_RESTORE_LOCK_PATH_ENV = "MEMPA_QDRANT_RESTORE_LOCK_PATH"


def qdrant_client(*, host: str, port: int) -> Any:
    """Build the one Qdrant client used by normal and admin operations."""
    from qdrant_client import QdrantClient

    return QdrantClient(
        host=host,
        port=port,
        timeout=QDRANT_OPERATION_TIMEOUT_SECONDS,
    )


@contextmanager
def qdrant_restore_lease() -> Iterator[Path]:
    """Serialize one complete restore against the shared Qdrant service.

    Only processes running this source acquire the lease. Older producers do
    not participate, so operators must not overlap their restores with these.
    """
    configured = os.environ.get(QDRANT_RESTORE_LOCK_PATH_ENV, "").strip()
    if not configured:
        raise RuntimeError(
            "Qdrant restore requires the shared service lock path: "
            f"{QDRANT_RESTORE_LOCK_PATH_ENV} is unset"
        )
    path = Path(configured).expanduser()
    if not path.is_absolute():
        raise ValueError(
            "Qdrant restore lock path must be absolute: "
            f"{QDRANT_RESTORE_LOCK_PATH_ENV}={configured!r}"
        )
    if not path.parent.is_dir():
        raise FileNotFoundError(
            "Qdrant restore lock parent does not exist: "
            f"path={str(path)!r} parent={str(path.parent)!r}"
        )
    with path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            yield path
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
