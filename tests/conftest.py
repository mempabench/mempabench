from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import pytest

from harness.run_layout import PROJECT_ROOT
from memory.backends.qdrant_runtime import QDRANT_RESTORE_LOCK_PATH_ENV
from mempabench.cli import PERSONA_DATA_FILES, load_persona_data_files


def _missing_persona_data(root: Path) -> list[str]:
    """The files of `PERSONA_DATA_FILES`, which `mempabench data pull` puts
    under the checkout root, that are not there, in persona order."""
    _, listed = load_persona_data_files(PERSONA_DATA_FILES)
    return [
        relative
        for files in listed.values()
        for relative in files
        if not (root / relative).is_file()
    ]


def pytest_sessionstart(session: pytest.Session) -> None:
    """Stop before any test runs when the persona data has not been pulled.

    Many tests read the four personas' data from the checkout root, where
    `mempabench data pull` puts it. This check never downloads it and never
    skips a test: without the data the run stops here and says what to do.
    """
    missing = _missing_persona_data(PROJECT_ROOT)
    if missing:
        shown = ", ".join(missing[:5]) + (", ..." if len(missing) > 5 else "")
        raise pytest.UsageError(
            f"persona data is missing under {PROJECT_ROOT}: {len(missing)} "
            f"files of {PERSONA_DATA_FILES.name} are absent ({shown}). Run `uv run mempabench data pull` "
            "in this checkout, then run the tests again."
        )


#: The vendored MemOS package writes `.memos/logs/` under this directory, or
#: under the working directory when it is unset, as soon as it is imported.
MEMOS_BASE_PATH_ENV = "MEMOS_BASE_PATH"
_memos_base_path_before: str | None = None
_memos_scratch: Path | None = None


def pytest_configure(config: pytest.Config) -> None:
    """Point MemOS at a scratch directory before any test module imports it, so
    the tests leave no `.memos/` in the checkout."""
    global _memos_base_path_before, _memos_scratch
    _memos_base_path_before = os.environ.get(MEMOS_BASE_PATH_ENV)
    _memos_scratch = Path(tempfile.mkdtemp(prefix="mempabench-tests-memos-"))
    os.environ[MEMOS_BASE_PATH_ENV] = str(_memos_scratch)


def pytest_unconfigure(config: pytest.Config) -> None:
    if _memos_base_path_before is None:
        os.environ.pop(MEMOS_BASE_PATH_ENV, None)
    else:
        os.environ[MEMOS_BASE_PATH_ENV] = _memos_base_path_before
    if _memos_scratch is not None:
        shutil.rmtree(_memos_scratch)


@pytest.fixture(scope="session", autouse=True)
def _shared_qdrant_restore_lock(
    tmp_path_factory: pytest.TempPathFactory,
):
    """Give restore tests one host-wide lock path unless they remove it."""
    previous = os.environ.get(QDRANT_RESTORE_LOCK_PATH_ENV)
    os.environ[QDRANT_RESTORE_LOCK_PATH_ENV] = str(
        tmp_path_factory.mktemp("qdrant-restore") / "service.lock"
    )
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(QDRANT_RESTORE_LOCK_PATH_ENV, None)
        else:
            os.environ[QDRANT_RESTORE_LOCK_PATH_ENV] = previous

