"""Action-type classification for MemPABench task tools.

Maps tool names to autonomy enforcement categories:
  read            — retrieval only, no side effects
  draft           — creates a local draft, not externally sent
  internal_write  — writes to run-local state (documents, calendar, contacts)
  external_action — sends, books, buys, pays, or files outside the PA's local scope
"""

from __future__ import annotations

from typing import Any

from pa.agent.ix.render_types import UNKNOWN_TOOL_ACTION_TYPE

_BASE_TOOL_ACTION_TYPES: dict[str, str] = {
    # read
    "document_read": "read",
    "email_message_search": "read",
    "email_message_read": "read",
    "email_draft_read": "read",
    "email_sent_read": "read",
    "message_search": "read",
    "message_read": "read",
    "phone_call_read": "read",
    "calendar_event_search": "read",
    "calendar_event_read": "read",
    "contact_search": "read",
    "contact_read": "read",
    "map_route_search": "read",
    "place_search": "read",
    "weather_forecast_read": "read",
    "shopping_product_search": "read",
    "shopping_order_read": "read",
    "package_tracking_read": "read",
    "utility_bill_read": "read",
    "payment_method_read": "read",
    "bill_payment_read": "read",
    "appointment_search": "read",
    "appointment_read": "read",
    "restaurant_reservation_search": "read",
    "restaurant_reservation_read": "read",
    "room_booking_search": "read",
    "room_booking_read": "read",
    "hotel_search": "read",
    "hotel_booking_read": "read",
    "flight_search": "read",
    "flight_read": "read",
    "flight_booking_read": "read",
    "scholarly_paper_search": "read",
    "scholarly_paper_read": "read",
    "recipe_search": "read",
    "reimbursement_submission_read": "read",
    "conference_registration_read": "read",
    "grade_submission_read": "read",
    "comic_store_inventory_read": "read",
    "cinema_showtime_search": "read",
    "cinema_ticket_read": "read",
    "exhibition_search": "read",
    "exhibition_ticket_read": "read",
    "media_transcode_service_search": "read",
    "media_transcode_job_read": "read",
    "household_inventory_search": "read",
    "club_closure_calendar_read": "read",
    "copier_entitlement_read": "read",
    "delivery_address_read": "read",
    "package_delivery_proof_read": "read",
    "shopping_return_case_read": "read",
    # draft
    "email_draft_create": "draft",
    # internal_write
    "document_create": "internal_write",
    "document_update": "internal_write",
    "document_delete": "internal_write",
    "calendar_event_create": "internal_write",
    "calendar_event_update": "internal_write",
    "contact_create": "internal_write",
    "contact_update": "internal_write",
    # external_action
    "email_message_send": "external_action",
    "message_send": "external_action",
    "phone_call_dial": "external_action",
    "shopping_order_create": "external_action",
    "shopping_order_update": "external_action",
    "bill_payment_create": "external_action",
    "appointment_create": "external_action",
    "appointment_update": "external_action",
    "restaurant_reservation_create": "external_action",
    "restaurant_reservation_update": "external_action",
    "room_booking_create": "external_action",
    "room_booking_update": "external_action",
    "hotel_book": "external_action",
    "hotel_booking_update": "external_action",
    "flight_book": "external_action",
    "reimbursement_submission_create": "external_action",
    "conference_registration_create": "external_action",
    "grade_submission_create": "external_action",
    "cinema_ticket_purchase": "external_action",
    "exhibition_ticket_purchase": "external_action",
    "media_transcode_job_create": "external_action",
    "corporate_reporting_submit": "external_action",
    "shopping_return_case_update": "external_action",
    "utility_account_holder_update": "external_action",
}

DEFAULT_TOOL_ACTION_TYPES: dict[str, str] = {
    **_BASE_TOOL_ACTION_TYPES,
    **{
        f"mcp_state_{name}": action_type
        for name, action_type in _BASE_TOOL_ACTION_TYPES.items()
    },
}


def tool_action_type(tool_name: str, overrides: dict[str, Any] | None = None) -> str:
    """Return the action type for a tool name.

    Checks overrides first, then DEFAULT_TOOL_ACTION_TYPES. An unregistered
    tool falls back to the most tightly gated class, not the least: "unknown"
    must never mean "exempt from the gate" (plan §3.8). The startup coverage
    check in `pa.agent.ix.collector.assert_action_type_coverage` is the main
    defence; this return value is the runtime floor behind it.
    """
    if overrides and tool_name in overrides:
        return str(overrides[tool_name])
    return DEFAULT_TOOL_ACTION_TYPES.get(tool_name, UNKNOWN_TOOL_ACTION_TYPE)
