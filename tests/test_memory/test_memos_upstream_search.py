"""Regression tests for MemOS behavior patched in the pinned submodule."""

from __future__ import annotations

import logging
from unittest.mock import Mock

import pytest

import memos.api.handlers  # noqa: F401 - initializes the upstream handler cycle
from memos.multi_mem_cube.single_cube import SingleCubeView
from memos.types.general_types import SearchMode


def _view() -> SingleCubeView:
    return SingleCubeView(
        cube_id="cube_test",
        naive_mem_cube=Mock(),
        mem_reader=Mock(),
        mem_scheduler=Mock(),
        logger=logging.getLogger("test.memos.search"),
        searcher=Mock(),
    )


def test_search_dependency_failure_propagates() -> None:
    view = _view()
    view._fast_search = Mock(side_effect=RuntimeError("qdrant unavailable"))

    with pytest.raises(RuntimeError, match="qdrant unavailable"):
        view._search_text(Mock(), Mock(), SearchMode.FAST)


def test_unsupported_search_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="value='unknown'"):
        _view()._search_text(Mock(), Mock(), "unknown")
