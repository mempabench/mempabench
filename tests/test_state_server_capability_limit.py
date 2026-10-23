"""The capability-limit signal on state read/retrieve tools.

Contract: only a read or
retrieve of an *already resolved* resource that the fixture explicitly marks
unavailable returns the capability-limit shape. A search with no hits stays an
ordinary empty result, and an identifier that matches no record stays the
existing not_found error.
"""

import json
from pathlib import Path
from typing import Any, get_args

import pytest

from pa.agent.ix import render_types
from pa.agent.ix.collector import is_capability_limit
from pa.agent.ix.registry import TurnToolRegistry
from pa.agent.ix.render_types import RequiredContext
from pa.agent.tools.registry import ToolRegistry
from state_server import store as store_module
from state_server.store import StateStore

FIXTURE_PAPERS = (
    Path(__file__).resolve().parents[1]
    / "data/state_fixtures/user_a/scholarly/papers.json"
)
ACC_026_PAPER_ID = "nakamura2026nonmarkov"
ACC_026_TITLE = "Non-Markovian corrections to fusion-channel dephasing near a gapless edge"


def _store(tmp_path: Path, session_date: str | None = None) -> StateStore:
    return StateStore(
        tmp_path / "state",
        run_id="run",
        user_id="user_a",
        session_id="acc_026",
        session_date=session_date,
    )


def _write(state: Path, relative: str, payload: Any) -> None:
    path = state / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _tool_log(store: StateStore) -> list[dict[str, Any]]:
    path = store.state_root / "tool_log.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


PAPERS = {
    "papers": [
        {
            "paper_id": "held2026blocked",
            "title": "Non-Markovian corrections near a gapless edge",
            "authors": ["R. Nakamura"],
            "abstract": "Corrections to fusion-channel dephasing.",
            "fulltext": "Full readable text that must never be served.",
            "unavailable": True,
            "boundary_kind": "missing_data",
            "unavailable_reason": "Metadata only; the full text is not in the catalog.",
        },
        {
            "paper_id": "open2025fine",
            "title": "Thermal noise in superconducting qubits",
            "authors": ["M. Ito"],
            "abstract": "Noise characterization for transmon devices.",
            "fulltext": "Full readable text of the thermal noise paper.",
        },
    ]
}

CONTACTS = {
    "contacts": [
        {
            "contact_id": "ct_0001",
            "name": "Prof. M. Reyes",
            "email": "m.reyes@stanmore.example",
            "unavailable": True,
            "boundary_kind": "external_dependency",
            "unavailable_reason": "The directory entry is held by the institution.",
        },
        {"contact_id": "ct_0002", "name": "Theo", "email": "theo@example.com"},
    ]
}

TRACKING = {
    "packages": [
        {
            "tracking_id": "CF881204",
            "carrier": "Coastline Freight",
            "events": [
                {"date": "2026-11-02", "status": "in_transit", "location": "Fresno, CA"},
                {"date": "2026-11-04", "status": "delivered", "location": "Glenmont, CA"},
            ],
            "unavailable": True,
            "boundary_kind": "external_dependency",
            "unavailable_reason": "The carrier will not release the delivery signature.",
        },
        {
            "tracking_id": "GLX240871",
            "carrier": "Glenmont Express",
            "events": [
                {"date": "2026-11-03", "status": "in_transit", "location": "Bakersfield, CA"},
            ],
        },
    ]
}


# --- the frozen shape -------------------------------------------------------


def test_duplicated_constants_equal_the_render_types_originals() -> None:
    """state_server/ imports nothing from pa/, so the four capability-limit
    literals are duplicated. They must stay equal to the values frozen in
    `render_types`."""
    assert store_module.CAPABILITY_LIMIT_ERROR_CODE == render_types.CAPABILITY_LIMIT_ERROR_CODE
    assert store_module.CAPABILITY_LIMIT_TOOL_STATUS == render_types.CAPABILITY_LIMIT_TOOL_STATUS
    assert store_module.CAPABILITY_LIMIT_RESULT_KEYS == render_types.CAPABILITY_LIMIT_RESULT_KEYS
    assert store_module.BOUNDARY_KINDS == get_args(render_types.BoundaryKind)
    # CapabilityLimitResult is those three keys plus the optional detail.
    assert tuple(render_types.CapabilityLimitResult.__annotations__) == (
        *store_module.CAPABILITY_LIMIT_RESULT_KEYS,
        "detail",
    )


def test_marked_paper_read_returns_the_frozen_shape(tmp_path: Path) -> None:
    state = tmp_path / "state"
    _write(state, "scholarly/papers.json", PAPERS)
    store = _store(tmp_path)

    result = store.scholarly_paper_read("held2026blocked")

    assert set(store_module.CAPABILITY_LIMIT_RESULT_KEYS) <= set(result)
    assert result["ok"] is False
    assert result["error_code"] == "capability_limit"
    assert result["boundary_kind"] == "missing_data"
    assert result["detail"] == "Metadata only; the full text is not in the catalog."
    # The withheld text never rides along.
    assert "paper" not in result
    assert "fulltext" not in json.dumps(result)

    entry = _tool_log(store)[-1]
    assert entry["tool"] == "scholarly_paper_read"
    assert entry["status"] == "capability_limit"
    assert entry["result_summary"] == {
        "error_code": "capability_limit",
        "boundary_kind": "missing_data",
    }


def test_marked_contact_read_returns_the_frozen_shape(tmp_path: Path) -> None:
    state = tmp_path / "state"
    _write(state, "contacts.json", CONTACTS)
    store = _store(tmp_path)

    result = store.contact_read("ct_0001")

    assert result["ok"] is False
    assert result["error_code"] == "capability_limit"
    assert result["boundary_kind"] == "external_dependency"
    assert _tool_log(store)[-1]["status"] == "capability_limit"


def test_marked_package_tracking_read_returns_the_frozen_shape(tmp_path: Path) -> None:
    """user_c W5 `w5_shipment_never_arrived`: the shipment the carrier says it
    delivered and will not reopen. The marked record reports the limit; the
    unmarked one beside it still returns its ordinary projection."""
    state = tmp_path / "state"
    _write(state, "shipping/tracking.json", TRACKING)
    store = _store(tmp_path)

    result = store.package_tracking_read("CF881204")

    assert result["ok"] is False
    assert result["error_code"] == "capability_limit"
    assert result["boundary_kind"] == "external_dependency"
    assert result["detail"] == "The carrier will not release the delivery signature."
    # The fixed projection never rides along with the limit.
    for key in ("carrier", "status", "events"):
        assert key not in result

    entry = _tool_log(store)[-1]
    assert entry["tool"] == "package_tracking_read"
    assert entry["status"] == "capability_limit"
    assert entry["result_summary"] == {
        "error_code": "capability_limit",
        "boundary_kind": "external_dependency",
    }

    plain = store.package_tracking_read("GLX240871")
    assert plain["carrier"] == "Glenmont Express"
    assert plain["status"] == "in_transit"
    assert len(plain["events"]) == 1
    assert _tool_log(store)[-1]["status"] == "ok"


def test_a_marker_without_a_valid_boundary_kind_fails_loudly(tmp_path: Path) -> None:
    state = tmp_path / "state"
    _write(
        state,
        "scholarly/papers.json",
        {"papers": [{"paper_id": "p1", "title": "t", "unavailable": True}]},
    )
    store = _store(tmp_path)

    with pytest.raises(ValueError, match="boundary_kind"):
        store.scholarly_paper_read("p1")


# --- what must NOT become a capability limit --------------------------------


def test_search_still_finds_the_marked_paper(tmp_path: Path) -> None:
    """The entry stays searchable so the model can resolve a precise id."""
    state = tmp_path / "state"
    _write(state, "scholarly/papers.json", PAPERS)
    store = _store(tmp_path)

    found = store.scholarly_paper_search("Non-Markovian corrections near a gapless edge")

    assert found["matches"][0]["paper_id"] == "held2026blocked"
    assert _tool_log(store)[-1]["status"] == "ok"
    # The fixture control keys are not part of the catalog metadata.
    summary = found["matches"][0]
    for key in ("unavailable", "boundary_kind", "unavailable_reason", "fulltext"):
        assert key not in summary


def test_contact_search_and_read_hide_the_marker_keys(tmp_path: Path) -> None:
    state = tmp_path / "state"
    _write(state, "contacts.json", CONTACTS)
    store = _store(tmp_path)

    found = store.lookup_contacts("Reyes")
    assert found["matches"][0]["contact_id"] == "ct_0001"
    for key in ("unavailable", "boundary_kind", "unavailable_reason"):
        assert key not in found["matches"][0]

    plain = store.contact_read("ct_0002")
    assert plain["contact"]["name"] == "Theo"


def test_a_search_with_no_hits_is_an_ordinary_empty_result(tmp_path: Path) -> None:
    state = tmp_path / "state"
    _write(state, "scholarly/papers.json", PAPERS)
    store = _store(tmp_path)

    empty = store.scholarly_paper_search("zzzz nothing matches this query zzzz")

    assert empty == {
        "query": "zzzz nothing matches this query zzzz",
        "matches": [],
    }
    entry = _tool_log(store)[-1]
    assert entry["status"] == "ok"
    assert entry["result_summary"] == {"matches": 0}


def test_an_unknown_identifier_is_still_the_old_not_found_error(tmp_path: Path) -> None:
    state = tmp_path / "state"
    _write(state, "scholarly/papers.json", PAPERS)
    _write(state, "contacts.json", CONTACTS)
    store = _store(tmp_path)

    with pytest.raises(FileNotFoundError):
        store.scholarly_paper_read("invented_by_the_model_2099")
    entry = _tool_log(store)[-1]
    assert entry["status"] == "error"
    assert entry["result_summary"] == {"error": "not_found"}

    with pytest.raises(FileNotFoundError):
        store.contact_read("ct_9999")
    assert _tool_log(store)[-1]["status"] == "error"


def test_unmarked_records_are_unaffected(tmp_path: Path) -> None:
    state = tmp_path / "state"
    _write(state, "scholarly/papers.json", PAPERS)
    store = _store(tmp_path)

    read = store.scholarly_paper_read("open2025fine")

    assert read["paper"]["paper_id"] == "open2025fine"
    assert "thermal noise" in read["paper"]["fulltext"]
    assert _tool_log(store)[-1]["status"] == "ok"


# --- the committed acc_026 fixture ------------------------------------------


def test_acc_026_paper_is_listed_but_not_retrievable(tmp_path: Path) -> None:
    """The real user_a fixture: the catalog lists the paper acc_026 asks for,
    and retrieving that resolved id reports the capability limit."""
    state = tmp_path / "state"
    (state / "scholarly").mkdir(parents=True, exist_ok=True)
    (state / "scholarly" / "papers.json").write_text(
        FIXTURE_PAPERS.read_text(encoding="utf-8"), encoding="utf-8"
    )
    store = _store(tmp_path, session_date="2026-10-15")

    found = store.scholarly_paper_search(ACC_026_TITLE)
    assert found["matches"][0]["paper_id"] == ACC_026_PAPER_ID

    blocked = store.scholarly_paper_read(ACC_026_PAPER_ID)
    assert blocked["ok"] is False
    assert blocked["error_code"] == "capability_limit"
    assert blocked["boundary_kind"] == "missing_data"
    assert _tool_log(store)[-1]["status"] == "capability_limit"


# --- the model-facing seam --------------------------------------------------


def test_the_public_store_hands_the_frozen_shape_through_intact(
    monkeypatch, tmp_path: Path
) -> None:
    """`_PublicStore` rebuilds every model-facing result; `ok` / `error_code`
    have to survive that, because the runner's whole capability path keys off
    `is_capability_limit(result)` and nothing else.

    Live runs read as if this seam had eaten the signal — the
    two forced settings rendered byte-identically with `gate: null`. It had
    not: `strip_run_identity` drops run identity and world-visibility keys only.
    This pins that, and follows the result all the way to
    `IX_capability_resolve` being required.
    """
    state = tmp_path / "state"
    _write(state, "scholarly/papers.json", PAPERS)
    monkeypatch.setenv("MEMPA_STATE_ROOT", str(state))
    monkeypatch.setenv("MEMPA_RUN_ID", "v3g07")
    monkeypatch.setenv("MEMPA_USER_ID", "user_a")
    monkeypatch.setenv("MEMPA_SESSION_ID", "acc_026")
    import state_server.server as server

    result = server._store().scholarly_paper_read("held2026blocked")

    # Still a dict, still carrying every frozen key.
    assert isinstance(result, dict)
    assert set(store_module.CAPABILITY_LIMIT_RESULT_KEYS) <= set(result)
    assert result["ok"] is False
    assert result["error_code"] == store_module.CAPABILITY_LIMIT_ERROR_CODE
    assert is_capability_limit(result)
    # The JSON the model actually receives reads the same way.
    assert is_capability_limit(json.dumps(result))

    # …and that is the only input the resolve tool's required rule needs.
    registry = TurnToolRegistry(
        ToolRegistry(), [], ix_context={"capability_limit_seen": True},
    )
    assert render_types.IX_CAPABILITY_RESOLVE_TOOL in (
        registry.required_ix_tool_names_for(
            RequiredContext(capability_limit_seen=True)
        )
    )
    # Without the flag the tool is not even offered, so nothing is forced.
    quiet = TurnToolRegistry(ToolRegistry(), [], ix_context={})
    assert render_types.IX_CAPABILITY_RESOLVE_TOOL not in (
        quiet.required_ix_tool_names_for(RequiredContext())
    )


PRODUCTS = {
    "currency": "USD",
    "products": [
        {
            "product_id": "shp_np_letterhead",
            "item_type": "letterhead",
            "name": "Cascade branch letterhead, 500 sheets (NP-L090)",
            "description": "Pre-printed branch letterhead, boxed in reams of 500.",
            "price": 64.0,
            "in_stock": False,
            "delivery_methods": ["home_delivery"],
            "unavailable": True,
            "boundary_kind": "external_dependency",
            "unavailable_reason": (
                "Back-ordered at the mill. Northline has no restock date."
            ),
        },
        {
            "product_id": "shp_umb_golf",
            "item_type": "umbrella",
            "name": "Golf Umbrella, 62-inch",
            "description": "Oversized two-person canopy with a fiberglass shaft.",
            "price": 41.0,
            "in_stock": False,
            "delivery_methods": ["home_delivery", "pickup"],
            "notes": "Currently out of stock; restock expected next month.",
        },
        {
            "product_id": "shp_umb_compact",
            "item_type": "umbrella",
            "name": "Compact Umbrella",
            "description": "Folds into a bag.",
            "price": 22.0,
            "in_stock": True,
            "delivery_methods": ["home_delivery"],
        },
    ],
}

ADDRESSES = {"addresses": [{"address_id": "addr_k4w8pv", "line1": "61 Larkspur St"}]}
PRODUCT_PAYMENT = {
    "payment_methods": [
        {"method_id": "pm_visa_4821", "kind": "card", "masked": "****4821"}
    ]
}


def _shopping_store(tmp_path: Path) -> StateStore:
    state = tmp_path / "state"
    _write(state, "shopping/products.json", PRODUCTS)
    _write(state, "shipping/addresses.json", ADDRESSES)
    _write(state, "finance/payment_methods.json", PRODUCT_PAYMENT)
    return _store(tmp_path)


def test_out_of_stock_products_are_searchable(tmp_path: Path) -> None:
    """`shp_umb_golf` exists in all four worlds and no search could reach it.

    Dropping out-of-stock records made the store lie about its catalog: the PA
    could not tell "we do not sell that" from "it is sold out", and an
    unavailable product could never be resolved to the id whose order raises
    the capability limit.
    """
    store = _shopping_store(tmp_path)

    result = store.shopping_product_search("umbrella")

    by_id = {record["product_id"]: record for record in result["matches"]}
    assert set(by_id) == {"shp_umb_golf", "shp_umb_compact"}
    assert by_id["shp_umb_golf"]["in_stock"] is False
    assert by_id["shp_umb_compact"]["in_stock"] is True

    # The fixture-only markers still never reach the model.
    letterhead = store.shopping_product_search("letterhead")["matches"][0]
    assert letterhead["product_id"] == "shp_np_letterhead"
    assert letterhead["in_stock"] is False
    for key in ("unavailable", "boundary_kind", "unavailable_reason"):
        assert key not in letterhead


def test_ordering_an_unavailable_product_is_a_formal_capability_limit(
    tmp_path: Path,
) -> None:
    """user_d acc_066: the promised branch letterhead is back-ordered at the mill.

    The outline wants a capability-limit ingredient, and the order path raised a
    plain ValueError indistinguishable from an ordinary stock-out.
    """
    store = _shopping_store(tmp_path)

    result = store.shopping_order_create(
        "shp_np_letterhead",
        quantity=1,
        delivery_address_id="addr_k4w8pv",
        payment_method_id="pm_visa_4821",
    )

    assert result["ok"] is False
    assert result["error_code"] == "capability_limit"
    assert result["boundary_kind"] == "external_dependency"
    assert result["detail"].startswith("Back-ordered at the mill")
    assert "order_id" not in result

    entry = _tool_log(store)[-1]
    assert entry["tool"] == "shopping_order_create"
    assert entry["status"] == "capability_limit"


def test_an_ordinary_stock_out_is_still_an_ordinary_error(tmp_path: Path) -> None:
    """`shp_umb_golf` is out of stock but not withheld: it restocks next month.

    The two must stay distinct -- one is a world boundary the PA has to work
    around, the other is a wait.
    """
    store = _shopping_store(tmp_path)

    with pytest.raises(ValueError, match="out of stock"):
        store.shopping_order_create(
            "shp_umb_golf",
            quantity=1,
            delivery_address_id="addr_k4w8pv",
            payment_method_id="pm_visa_4821",
        )

    with pytest.raises(ValueError, match="unknown product"):
        store.shopping_order_create(
            "shp_no_such_thing",
            quantity=1,
            delivery_address_id="addr_k4w8pv",
            payment_method_id="pm_visa_4821",
        )


def test_the_user_d_fixture_letterhead_yields_the_limit(tmp_path: Path) -> None:
    """The real acc_066 ingredient, read from user_d's own fixture."""
    fixture = (
        Path(__file__).resolve().parents[1]
        / "data/state_fixtures/user_d/shopping/products.json"
    )
    record = next(
        product
        for product in json.loads(fixture.read_text(encoding="utf-8"))["products"]
        if product["product_id"] == "shp_np_letterhead"
    )
    assert record["unavailable"] is True
    assert record["boundary_kind"] == "external_dependency"

    state = tmp_path / "state"
    _write(state, "shopping/products.json", {"products": [record]})
    _write(state, "shipping/addresses.json", ADDRESSES)
    _write(state, "finance/payment_methods.json", PRODUCT_PAYMENT)
    store = _store(tmp_path)

    result = store.shopping_order_create(
        "shp_np_letterhead",
        quantity=1,
        delivery_address_id="addr_k4w8pv",
        payment_method_id="pm_visa_4821",
    )
    assert result["error_code"] == "capability_limit"
    assert result["boundary_kind"] == "external_dependency"
