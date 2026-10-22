"""`test_resume_pool.py` is skipped on this branch. This proves it still says something.

Plan §9 step 9's acceptance suite waits for `memory/turn_journal.py`, which is
step 9's own deliverable, so on this branch the suite is skipped at import and
contributes no red. That is an honest state to be in, but only if two claims
about the skipped file can be checked rather than taken on trust:

1. **it is satisfiable** -- no assertion in it is an equality the design
   rejects, so a correct step 9 can turn all three tests green. A test that can
   never pass is worse than no test: under `strict=True` it never fires;
2. **it is non-vacuous** -- the tests fail on their own assertions, not on the
   module's absence, and specifically they refuse the shortcut §6.1 (b) rules
   out (a journal that keeps a span and a hash rather than the pool itself).

Both are checked here by loading the suite twice, once against each throwaway
journal in `_stubs/`, and running its three tests by hand. This file runs
today; when step 9 lands `memory/turn_journal.py` the suite itself runs for
real and this proof becomes redundant -- delete it in that commit.
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
import traceback
from pathlib import Path
from types import ModuleType

import pytest

SUITE_PATH = Path(__file__).with_name("test_resume_pool.py")
STUB_DIR = Path(__file__).with_name("_stubs")
TEST_NAMES = (
    "test_rederived_pool_equals_the_uninterrupted_pool",
    "test_resume_writes_pools_after_the_committed_turn_and_none_before",
    "test_crash_between_adapter_ack_and_cursor_save_replays_and_logs",
)
#: The assertion each test must fail on when the journal loses the pool -- the
#: rendered pool's byte-equality, the prose surviving a replay, and the resume
#: log line §9 step 9's verification column asks for.
REFUTED_BY = (
    "assert rederived.messages == reference.messages",
    "assert INTERMEDIATE_PROSE in text",
    'assert any("turn_01" in line for line in lines)',
)


def _load(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _stub(tag: str) -> ModuleType:
    """One throwaway journal, loaded by path so `tests/` need not be a package."""
    return _load(STUB_DIR / f"{tag}_turn_journal.py", f"_step9_stub_{tag}")


def _load_suite(monkeypatch: pytest.MonkeyPatch, stub: ModuleType, tag: str) -> ModuleType:
    """Import `test_resume_pool.py` with `memory.turn_journal` bound to `stub`.

    `spec_from_file_location` with a private name, rather than an import by
    package path: the suite is loaded twice in one process, against two
    different journals, and neither load may see the other's module object.
    """
    monkeypatch.setitem(sys.modules, "memory.turn_journal", stub)
    spec = importlib.util.spec_from_file_location(f"_step9_suite_{tag}", SUITE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def _run(module: ModuleType, name: str, tmp_path: Path) -> None:
    """Run one acceptance test to completion, with its own fixtures."""
    with pytest.MonkeyPatch.context() as inner:
        asyncio.run(getattr(module, name)(tmp_path=tmp_path, monkeypatch=inner))


def test_the_step9_suite_passes_against_a_faithful_journal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Claim 1: every assertion in the suite is satisfiable.

    The faithful stub is the smallest journal §6.1 (b) describes -- it persists
    the rendered pool per `turn_id`, and its `replay_pending_turns` rewrites
    every pending record and logs the ids. Against it all three tests pass, so
    none of them is asserting an equality the plan rejects, and each of the
    three `strict=True` xfails can actually fire once step 9 lands.
    """
    module = _load_suite(monkeypatch, _stub("faithful"), "faithful")
    for index, name in enumerate(TEST_NAMES):
        _run(module, name, tmp_path / f"faithful_{index}")


@pytest.mark.parametrize(("name", "expected_line"), zip(TEST_NAMES, REFUTED_BY))
def test_the_step9_suite_fails_against_a_journal_that_loses_the_pool(
    name: str, expected_line: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Claim 2: the suite refuses the shortcut, and fails by assertion.

    The inert stub keeps the span and a payload hash and throws the pool away,
    which is the alternative §6.1 (b) rejects, and its replay logs nothing. The
    failure must be an `AssertionError`: an `ImportError`, a `TypeError` or an
    `AttributeError` here would mean the suite is only checking that a module
    with the right names exists.

    And it must be the *right* assertion. `pytest.raises(AssertionError)` alone
    would be satisfied by any incidental failure anywhere in the test, so the
    frame that raised is read back and matched against the line named in
    `REFUTED_BY`: the pool byte-equality, the intermediate prose surviving a
    replay, and the resume's own log line -- one property each.
    """
    module = _load_suite(monkeypatch, _stub("inert"), "inert")
    with pytest.raises(AssertionError) as caught:
        _run(module, name, tmp_path)

    frames = [
        frame for frame in traceback.extract_tb(caught.tb)
        if Path(frame.filename) == SUITE_PATH
    ]
    assert frames, f"{name} failed outside the suite itself"
    assert frames[-1].line == expected_line, (
        f"{name} failed at {frames[-1].line!r}, not at the assertion the inert "
        f"journal is meant to refute"
    )
