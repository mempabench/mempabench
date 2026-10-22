"""The outline schema's toolbox-domain enum must match the tool catalog."""

from pathlib import Path
from typing import get_args

import yaml

from scripts.outline_schema import ToolboxDomain


def test_schema_domains_match_catalog_domains():
    catalog = yaml.safe_load(
        Path("data/state_tools/catalog.yaml").read_text(encoding="utf-8")
    )
    catalog_domains = {spec["domain"] for spec in catalog["tools"].values()}
    schema_domains = set(get_args(ToolboxDomain))
    assert schema_domains == catalog_domains


def test_catalog_handlers_are_all_real():
    catalog = yaml.safe_load(
        Path("data/state_tools/catalog.yaml").read_text(encoding="utf-8")
    )
    for name, spec in catalog["tools"].items():
        assert spec["status"] == "implemented", name
        assert spec["implementation"] == f"state_server.server:{name}", name
        assert spec["backing_stores"], name
        assert spec["layer"] == "core" or spec["layer"].startswith("persona:"), name


def test_every_catalog_tool_has_a_server_function(monkeypatch, tmp_path):
    monkeypatch.setenv("MEMPA_STATE_ROOT", str(tmp_path / "state"))
    import state_server.server as server

    catalog = yaml.safe_load(
        Path("data/state_tools/catalog.yaml").read_text(encoding="utf-8")
    )
    for name in catalog["tools"]:
        assert callable(getattr(server, name, None)), name
