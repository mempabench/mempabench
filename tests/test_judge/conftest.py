"""`--update-snapshots`: rewrite `tests/test_judge/snapshots/*.md` instead of comparing.

    pytest tests/test_judge --update-snapshots

The option is registered here, so it is parsed only when `tests/test_judge`
(or a path under it) is on the command line; a full-suite run neither needs
nor accepts it.
"""

from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--update-snapshots", action="store_true", default=False,
        help="rewrite the judge prompt snapshots from the current assembly instead of comparing",
    )


@pytest.fixture
def update_snapshots(request: pytest.FixtureRequest) -> bool:
    return bool(request.config.getoption("--update-snapshots"))
