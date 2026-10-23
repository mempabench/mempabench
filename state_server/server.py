"""MCP entrypoint for MemPABench task-state tools.

Every tool is a real handler over the run-local state store. There is no
scripted-case layer: results come from store files, and writes persist.
"""

from __future__ import annotations

from typing import Annotated, Any, Callable

from mcp.server.fastmcp import FastMCP
from pydantic import Field

from state_server.store import StateStore, strip_run_identity

mcp = FastMCP("mempabench-state")


# Parameter formats the PA can see. Every date, time and closed vocabulary below
# is later compared as a string or parsed by `state_server/store.py`, so an
# undocumented format was a silent mismatch: an empty search, or a stored record
# no later read could find. The store now rejects a bad value and these
# descriptions say what a good one looks like -- keep the two in step.


class _PublicStore:
    """The store as the model sees it: every result minus the run bookkeeping.

    `StateStore` stamps `run_id` / `user_id` / `session_id` onto the records it
    creates at ~18 sites, and a read hands the whole record back -- so a PA that
    created an appointment and read it back received `"run_id": "v2g03"` in its
    own prompt. The stamp stays on disk and in `tool_log.jsonl`, where run
    attribution belongs; this is the seam where every model-facing result leaves,
    so it is where the stamp comes off. Nothing else changes: the wrapped method
    is called exactly as before, and errors propagate untouched.
    """

    def __init__(self, inner: StateStore) -> None:
        self._inner = inner

    def __getattr__(self, name: str) -> Callable[..., Any]:
        method = getattr(self._inner, name)

        def _call(*args: Any, **kwargs: Any) -> Any:
            return strip_run_identity(method(*args, **kwargs))

        return _call


def _store() -> _PublicStore:
    return _PublicStore(StateStore.from_env())


# Documents

@mcp.tool(
    name="copier_entitlement_read",
    description=(
        "Read one copier-contract entitlement by contract and component. A schedule "
        "the provider keeps in its portal returns the standard capability-limit shape."
    ),
)
def copier_entitlement_read(
    contract_id: Annotated[str, Field(description="Copier service contract id, ctr_*.")],
    component: Annotated[str, Field(description="Covered component, such as imaging drum.")],
) -> dict:
    return _store().copier_entitlement_read(contract_id, component)


@mcp.tool(
    name="corporate_reporting_submit",
    description=(
        "Submit one ready run-state file through its configured corporate reporting "
        "account. This is an external action. A locked account returns the standard "
        "capability-limit shape and creates no submission."
    ),
)
def corporate_reporting_submit(
    file_id: Annotated[
        str,
        Field(description="Run-state-relative path, such as documents/branch/monthly.csv."),
    ],
) -> dict:
    return _store().corporate_reporting_submit(file_id)

@mcp.tool(
    name="document_read",
    description=(
        "Read a document from the current MemPABench run state. Use this for task documents, "
        "not for interaction preferences. Paths are relative to the user's task state: an "
        "empty path or '.' lists the top-level folders, a user reference like 'my desktop' "
        "corresponds to the my_desktop/ folder, and the user's working files (for example "
        "'the research folder' or 'the admin folder') live under documents/. Use a path the "
        "user supplied or a prior task result surfaced. Do not browse unrelated directories "
        "to discover task inputs the user has not identified. This state holds only the "
        "user's task documents; your own memory and skills files are not here, so paths "
        "like memory/MEMORY.md are not readable through this tool."
    ),
)
def document_read(
    path: Annotated[
        str,
        Field(
            description=(
                "Document path relative to the task state, such as "
                "'documents/research/draft_intro.md' or 'my_desktop/notes.md'. "
                "Empty or '.' lists the top-level folders."
            )
        ),
    ],
) -> dict:
    return _store().read_document(path)


@mcp.tool(
    name="document_create",
    description=(
        "Create a new document in the current MemPABench run state. Use this only "
        "when the task needs a file that does not exist yet. If the path already "
        "exists nothing is written and the result reports status 'exists'; edit "
        "that file with document_update instead."
    ),
)
def document_create(
    path: Annotated[
        str,
        Field(
            description=(
                "Path for the new document, relative to the task state, "
                "including the file extension."
            )
        ),
    ],
    content: Annotated[
        str,
        Field(
            description=(
                "Full text of the new document. When this is something the user "
                "must dictate, pass their own words; if the user has not given "
                "them, ask the user instead of composing it."
            )
        ),
    ],
) -> dict:
    return _store().create_document(path, content)


@mcp.tool(
    name="document_update",
    description=(
        "Edit an existing document in the current MemPABench run state. 'anchor' is "
        "an exact substring of the document's current text that must occur exactly "
        "once; it is replaced by 'replacement'. Read the document first and copy the "
        "anchor as plain text: document_read returns JSON, so the "
        r"\n and \" sequences in its output are JSON escapes, not document text — "
        "send real newlines and plain quotes, never a backslash. Prefer a short "
        "single-line anchor, and do not let one span a blank line. "
        "The rest of the file is left untouched. Fails if the file "
        "does not exist or the anchor is missing or ambiguous. This state holds only "
        "the user's task documents; your own memory and skills files are not here, so "
        "paths like memory/MEMORY.md are not writable through this tool."
    ),
)
def document_update(
    path: Annotated[
        str, Field(description="Path of the existing document to edit.")
    ],
    anchor: Annotated[
        str,
        Field(
            description=(
                "Exact substring of the document's current text, copied "
                "verbatim, occurring exactly once. Send it as plain text: real "
                "newlines and plain quotes, never the "
                r"\n or \" JSON escapes that appear in document_read output. "
                "Keep it short, ideally a single line, and do not let it span "
                "a blank line. Never a bare heading line on its own — anchor on "
                "a content line under the heading, or on the heading plus its "
                "first content line."
            )
        ),
    ],
    replacement: Annotated[
        str,
        Field(
            description=(
                "Text that replaces the anchor. When this is something the user "
                "must dictate, pass their own words; if the user has not given "
                "them, ask the user instead of composing it. An empty string "
                "deletes the anchored text; use it only to remove a whole entry "
                "or line you have read back, and never leave an unmatched fence "
                "or heading behind."
            )
        ),
    ],
) -> dict:
    return _store().update_document(path, anchor, replacement)


@mcp.tool(
    name="document_delete",
    description=(
        "Delete an existing document from the current MemPABench run state. Use this "
        "only when the user's task calls for removing a file. Fails if it is missing."
    ),
)
def document_delete(
    path: Annotated[
        str, Field(description="Path of the existing document to delete.")
    ],
) -> dict:
    return _store().delete_document(path)


# Email (formal channel)

@mcp.tool(
    name="email_message_search",
    description=(
        "Search the run-local email inbox by keyword. Use this when the task asks "
        "the PA to find an email or recover details from email."
    ),
)
def email_message_search(
    query: Annotated[
        str,
        Field(
            description=(
                "Free-text keywords matched against sender, subject and body; "
                "an unmatched word returns no results rather than an error."
            )
        ),
    ],
) -> dict:
    return _store().search_email(query)


@mcp.tool(
    name="email_message_read",
    description=(
        "Read the full body of an inbox email by id after email_message_search "
        "returns a match."
    ),
)
def email_message_read(
    email_id: Annotated[
        str,
        Field(description="Inbox email id as returned by email_message_search, em_NNNN."),
    ],
) -> dict:
    return _store().read_email(email_id)


@mcp.tool(
    name="email_draft_create",
    description=(
        "Save a PA-authored email draft. Pass draft_id to revise an existing draft "
        "in place; omit it to create a new draft. Optional attachments must be "
        "run-state-relative paths."
    ),
)
def email_draft_create(
    to: Annotated[
        str,
        Field(
            description=(
                "One recipient, or multiple recipients separated by commas, each "
                "as an email address or a name; stored verbatim and never resolved "
                "against the address book."
            )
        ),
    ],
    subject: Annotated[str, Field(description="Subject line.")],
    body: Annotated[
        str,
        Field(
            description=(
                "Full body text of the draft. When this is something the user "
                "must dictate, pass their own words; if the user has not given "
                "them, ask the user instead of composing it."
            )
        ),
    ],
    attachments: Annotated[
        list[str] | None,
        Field(description="Document paths relative to the task state."),
    ] = None,
    draft_id: Annotated[
        str | None,
        Field(
            description=(
                "Draft id as returned by an earlier email_draft_create, "
                "draft_NNNN; omit to create a new draft."
            )
        ),
    ] = None,
) -> dict:
    return _store().save_email_draft(to, subject, body, attachments, draft_id)


@mcp.tool(
    name="email_draft_read",
    description="Read one saved email draft by identifier.",
)
def email_draft_read(
    draft_id: Annotated[
        str,
        Field(description="Draft id as returned by email_draft_create, draft_NNNN."),
    ],
) -> dict:
    return _store().read_email_draft(draft_id)


@mcp.tool(
    name="email_message_send",
    description=(
        "Send an email from the current MemPABench run state. This is an external "
        "action: use it only when the user has authorized sending. Optional attachments "
        "must be run-state-relative paths."
    ),
)
def email_message_send(
    to: Annotated[
        str,
        Field(
            description=(
                "Recipient, as an email address or a name; stored verbatim and "
                "never resolved against the address book."
            )
        ),
    ],
    subject: Annotated[str, Field(description="Subject line.")],
    body: Annotated[
        str,
        Field(
            description=(
                "Full body text of the email. When this is something the user "
                "must dictate, pass their own words; if the user has not given "
                "them, ask the user instead of composing it."
            )
        ),
    ],
    attachments: Annotated[
        list[str] | None,
        Field(description="Document paths relative to the task state."),
    ] = None,
) -> dict:
    return _store().send_email(to, subject, body, attachments)


@mcp.tool(
    name="email_sent_read",
    description="Read one persisted sent-email record by identifier.",
)
def email_sent_read(
    sent_id: Annotated[
        str,
        Field(description="Sent-email id as returned by email_message_send, sent_NNNN."),
    ],
) -> dict:
    return _store().read_sent_email(sent_id)


# Messages (informal channel)

@mcp.tool(
    name="message_send",
    description=(
        "Send an informal text message to a personal contact. This is an external "
        "action: use it only when appropriate and authorized. Use email for formal "
        "or institutional correspondence."
    ),
)
def message_send(
    to: Annotated[
        str,
        Field(
            description=(
                "Contact name or handle; stored verbatim and never resolved "
                "against the address book."
            )
        ),
    ],
    body: Annotated[
        str,
        Field(
            description=(
                "Message text. When this is something the user must dictate, "
                "pass their own words; if the user has not given them, ask the "
                "user instead of composing it."
            )
        ),
    ],
) -> dict:
    return _store().message_send(to, body)


@mcp.tool(
    name="message_search",
    description="Search text-message history by keyword or contact.",
)
def message_search(
    query: Annotated[
        str,
        Field(
            description=(
                "Free-text keywords matched against the contact and the message "
                "text; an unmatched word returns no results rather than an error."
            )
        ),
    ],
) -> dict:
    return _store().message_search(query)


@mcp.tool(
    name="message_read",
    description="Read one text message by identifier.",
)
def message_read(
    message_id: Annotated[
        str,
        Field(
            description=(
                "Message id as returned by message_search or message_send, "
                "msg_NNNN."
            )
        ),
    ],
) -> dict:
    return _store().message_read(message_id)


# Calendar

@mcp.tool(
    name="calendar_event_search",
    description=(
        "Search the current run calendar by task, participant, or purpose. Results "
        "include personal events and world events such as holidays."
    ),
)
def calendar_event_search(
    query: Annotated[
        str,
        Field(
            description=(
                "Free-text keywords matched against title, location, notes and "
                "participants; an unmatched word returns no results rather than "
                "an error."
            )
        ),
    ],
    date: Annotated[
        str | None, Field(description="Optional date filter, YYYY-MM-DD.")
    ] = None,
) -> dict:
    return _store().calendar_event_search(query, date)


@mcp.tool(
    name="calendar_event_read",
    description="Read one calendar event by identifier.",
)
def calendar_event_read(
    event_id: Annotated[
        str,
        Field(description="Event id as returned by calendar_event_search, ev_NNNN."),
    ],
) -> dict:
    return _store().calendar_event_read(event_id)


@mcp.tool(
    name="calendar_event_create",
    description="Create one calendar event or recurring series.",
)
def calendar_event_create(
    title: Annotated[str, Field(description="Event title as it appears on the calendar.")],
    date: Annotated[
        str,
        Field(description="Event date, YYYY-MM-DD (first occurrence of a series)."),
    ],
    start_time: Annotated[
        str | None, Field(description="Start time, 24-hour HH:MM.")
    ] = None,
    end_time: Annotated[
        str | None, Field(description="End time, 24-hour HH:MM.")
    ] = None,
    location: Annotated[
        str | None, Field(description="Free-text place, such as a room or address.")
    ] = None,
    participants: Annotated[
        list[str] | None,
        Field(
            description=(
                "Names or email addresses; stored verbatim and never resolved "
                "against the address book."
            )
        ),
    ] = None,
    recurrence: Annotated[
        str | None,
        Field(
            description=(
                "Recurrence phrase: 'daily', 'yearly', or 'weekly on <day names>' "
                "such as 'weekly on Mondays and Wednesdays', each optionally "
                "suffixed ' through YYYY-MM-DD'. For a one-off event omit this "
                "parameter or pass null."
            )
        ),
    ] = None,
) -> dict:
    return _store().calendar_event_create(
        title, date, start_time, end_time, location, participants, recurrence
    )


@mcp.tool(
    name="calendar_event_update",
    description=(
        "Update or cancel one existing personal calendar event. World events such "
        "as holidays cannot be changed."
    ),
)
def calendar_event_update(
    event_id: Annotated[
        str,
        Field(description="Event id as returned by calendar_event_search, ev_NNNN."),
    ],
    title: Annotated[str | None, Field(description="New event title.")] = None,
    date: Annotated[str | None, Field(description="New date, YYYY-MM-DD.")] = None,
    start_time: Annotated[
        str | None, Field(description="New start time, 24-hour HH:MM.")
    ] = None,
    end_time: Annotated[
        str | None, Field(description="New end time, 24-hour HH:MM.")
    ] = None,
    location: Annotated[
        str | None, Field(description="New free-text place, such as a room or address.")
    ] = None,
    participants: Annotated[
        list[str] | None,
        Field(
            description=(
                "Replacement list of names or email addresses; stored verbatim "
                "and never resolved against the address book."
            )
        ),
    ] = None,
    status: Annotated[
        str | None, Field(description="Either 'confirmed' or 'cancelled'.")
    ] = None,
    notes: Annotated[
        str | None,
        Field(
            description=(
                "New free-text note kept on the event, such as what changed and "
                "why; omitted leaves the existing note in place."
            )
        ),
    ] = None,
) -> dict:
    return _store().calendar_event_update(
        event_id, title, date, start_time, end_time, location, participants, status,
        notes,
    )


# Contacts

@mcp.tool(
    name="contact_search",
    description=(
        "Look up contacts or external handoff channels by query. Use this when a "
        "task needs the right person, office, or channel to continue."
    ),
)
def contact_search(
    query: Annotated[
        str,
        Field(
            description=(
                "Free-text keywords matched against name, organization, relation "
                "and notes; an unmatched word returns no results rather than an "
                "error."
            )
        ),
    ],
) -> dict:
    return _store().lookup_contacts(query)


@mcp.tool(
    name="contact_read",
    description=(
        "Read one contact by identifier. A contact the address book lists but "
        'cannot serve returns {"ok": false, "error_code": "capability_limit", '
        '"boundary_kind": ...} instead of the record; an identifier that matches '
        "no contact is still an error."
    ),
)
def contact_read(
    contact_id: Annotated[
        str,
        Field(description="Contact id as returned by contact_search, ct_NNNN."),
    ],
) -> dict:
    return _store().contact_read(contact_id)


@mcp.tool(
    name="contact_create",
    description="Save a new contact in the current run state.",
)
def contact_create(
    name: Annotated[str, Field(description="Full name of the contact.")],
    email: Annotated[str | None, Field(description="Email address.")] = None,
    phone: Annotated[str | None, Field(description="Phone number.")] = None,
    organization: Annotated[
        str | None, Field(description="Employer, department or institution.")
    ] = None,
    relation: Annotated[
        str | None,
        Field(description="How the user knows them, such as 'collaborator' or 'landlord'."),
    ] = None,
    notes: Annotated[str | None, Field(description="Free-text notes.")] = None,
) -> dict:
    return _store().contact_create(name, email, phone, organization, relation, notes)


@mcp.tool(
    name="contact_update",
    description="Update one existing contact by identifier.",
)
def contact_update(
    contact_id: Annotated[
        str,
        Field(description="Contact id as returned by contact_search, ct_NNNN."),
    ],
    name: Annotated[str | None, Field(description="New full name.")] = None,
    email: Annotated[str | None, Field(description="New email address.")] = None,
    phone: Annotated[str | None, Field(description="New phone number.")] = None,
    organization: Annotated[
        str | None, Field(description="New employer, department or institution.")
    ] = None,
    relation: Annotated[
        str | None,
        Field(description="New relationship, such as 'collaborator' or 'landlord'."),
    ] = None,
    notes: Annotated[str | None, Field(description="New free-text notes.")] = None,
) -> dict:
    return _store().contact_update(
        contact_id, name, email, phone, organization, relation, notes
    )


# Phone

@mcp.tool(
    name="phone_call_dial",
    description=(
        "Place a phone call to a number this world lists, for a contact or a place. "
        "This is an external action: use it only when appropriate and authorized. "
        "Get the number from contact_read or place_search first and pass it back as "
        "the listing writes it. A line the world cannot put through returns "
        '{"ok": false, "error_code": "capability_limit", "boundary_kind": ...} '
        "instead of a call; a number no listing carries is an error. When the line "
        "answers, the result reports who answered and what they said."
    ),
)
def phone_call_dial(
    number: Annotated[
        str,
        Field(
            description=(
                "The number to dial, as the contact or place listing writes it; "
                "punctuation and spacing are ignored, so '(555) 026-2200' and "
                "'555-026-2200' reach the same line. A number no listing in this "
                "world carries is an error, not an unanswered call."
            )
        ),
    ],
    purpose: Annotated[
        str,
        Field(
            description=(
                "What this call is about, said to whoever answers. When this is "
                "something the user must dictate, pass their own words; if the "
                "user has not given them, ask the user instead of composing it."
            )
        ),
    ],
) -> dict:
    return _store().phone_call_dial(number, purpose)


@mcp.tool(
    name="phone_call_read",
    description="Read one call this run placed, by identifier.",
)
def phone_call_read(
    call_id: Annotated[
        str,
        Field(description="Call id as returned by phone_call_dial, call_NNNN."),
    ],
) -> dict:
    return _store().phone_call_read(call_id)


# Maps and places

@mcp.tool(
    name="club_closure_calendar_read",
    description=(
        "Read a club's dated closure calendar. A member-only calendar returns the "
        "standard capability-limit shape; ordinary weekly hours remain available "
        "through place_search."
    ),
)
def club_closure_calendar_read(
    place_id: Annotated[str, Field(description="Stable place identifier from the world record, place_*.")],
    date_range: Annotated[str, Field(description="Requested date range, YYYY-MM-DD/YYYY-MM-DD.")],
) -> dict:
    return _store().club_closure_calendar_read(place_id, date_range)

@mcp.tool(
    name="map_route_search",
    description=(
        "Return route options with travel time and estimated price for a trip, "
        "optionally through an intermediate stop."
    ),
)
def map_route_search(
    origin: Annotated[
        str,
        Field(
            description=(
                "Start of the trip as a place name or address; matched against "
                "the route fixture by text, so use a name a place_search result "
                "or the user gave."
            )
        ),
    ],
    destination: Annotated[
        str, Field(description="End of the trip, named the same way as origin.")
    ],
    stop: Annotated[
        str,
        Field(description="Optional intermediate stop, named the same way; '' for none."),
    ] = "",
) -> dict:
    return _store().route_options(origin, destination, stop)


@mcp.tool(
    name="place_search",
    description=(
        "Return nearby place options by location, category or name. Give at "
        "least one of them; they combine. Place records carry their own "
        "business hours, including dated holiday notices."
    ),
)
def place_search(
    location: Annotated[
        str | None,
        Field(
            description=(
                "Optional street or landmark near the user as the world names "
                "it, such as 'Civic Plaza' or 'Alder St', never a business or "
                "person name; an unknown value is a tool error that lists the "
                "known ones."
            )
        ),
    ] = None,
    category: Annotated[
        str | None,
        Field(
            description=(
                "Optional kind of place, matched against each record's category "
                "and tags. A category is one lowercase word, such as "
                "restaurant, cafe, grocery, cinema, exhibition, clinic or "
                "pharmacy; a world lists only the categories it actually "
                "contains, and an unknown value is a tool error that lists the "
                "known ones."
            )
        ),
    ] = None,
    query: Annotated[
        str | None,
        Field(
            description=(
                "Optional business or place name, such as 'Glenmont Pharmacy'; "
                "every whitespace-separated word has to appear in the listing's "
                "own name, case-insensitively. Use this when the user named the "
                "place; a name the world does not list simply returns no "
                "matches."
            )
        ),
    ] = None,
    limit: Annotated[int, Field(description="Maximum number of places to return.")] = 3,
) -> dict:
    return _store().place_search(location, category, query, limit)


# Weather

@mcp.tool(
    name="weather_forecast_read",
    description=(
        "Read the forecast for one location, date, and time window from the "
        "current run state."
    ),
)
def weather_forecast_read(
    location: Annotated[
        str,
        Field(
            description=(
                "City or neighbourhood the forecast records name, such as "
                "'Glenmont, CA'; an unknown value is a tool error that lists "
                "the known ones."
            )
        ),
    ],
    date: Annotated[str, Field(description="Forecast date, YYYY-MM-DD.")],
    time_window: Annotated[
        str,
        Field(
            description=(
                "Window as HH:MM-HH:MM, or one of morning / midday / afternoon / "
                "evening / tonight / night."
            )
        ),
    ],
) -> dict:
    return _store().weather_forecast_read(location, date, time_window)


# Shopping

@mcp.tool(
    name="shopping_product_search",
    description=(
        "Search the product catalog by item, with optional price and delivery "
        "constraints. Out-of-stock products are listed too, with in_stock "
        "false, so a sold-out item can be told apart from one the shop does "
        "not carry."
    ),
)
def shopping_product_search(
    query: Annotated[
        str,
        Field(
            description=(
                "Free-text keywords matched against product name, brand and "
                "tags; an unmatched word returns no results rather than an error."
            )
        ),
    ],
    max_price: Annotated[
        float | None, Field(description="Optional price ceiling in dollars.")
    ] = None,
    delivery_method: Annotated[
        str | None, Field(description="Either 'home_delivery' or 'pickup'.")
    ] = None,
) -> dict:
    return _store().shopping_product_search(query, max_price, delivery_method)


@mcp.tool(
    name="shopping_order_create",
    description=(
        "Place an authorized product order using a saved delivery address and "
        "payment method. A product that is out of stock cannot be ordered. "
        "This is an external action."
    ),
)
def shopping_order_create(
    product_id: Annotated[
        str,
        Field(description="Product id as returned by shopping_product_search, shp_*."),
    ],
    quantity: Annotated[int, Field(description="Number of units to order, at least 1.")],
    delivery_address_id: Annotated[
        str,
        Field(
            description=(
                "Saved delivery address id from delivery_address_read, addr_*."
            )
        ),
    ],
    payment_method_id: Annotated[
        str,
        Field(description="Saved payment method id from payment_method_read, pm_*."),
    ],
) -> dict:
    return _store().shopping_order_create(
        product_id, quantity, delivery_address_id, payment_method_id
    )


@mcp.tool(
    name="shopping_order_read",
    description="Read one persisted product order for verification.",
)
def shopping_order_read(
    order_id: Annotated[
        str,
        Field(
            description=(
                "Order id in the merchant's own format: a two-letter merchant "
                "code, a hyphen, six digits, such as AB-118427. Returned by "
                "shopping_order_create and shown on every order in the history."
            )
        ),
    ],
) -> dict:
    return _store().shopping_order_read(order_id)


@mcp.tool(
    name="shopping_order_update",
    description=(
        "Change or cancel one existing order that has not shipped. This is an "
        "external action."
    ),
)
def shopping_order_update(
    order_id: Annotated[
        str,
        Field(
            description=(
                "Order id in the merchant's own format: a two-letter merchant "
                "code, a hyphen, six digits, such as AB-118427. Returned by "
                "shopping_order_create and shown on every order in the history."
            )
        ),
    ],
    status: Annotated[
        str | None, Field(description="Either 'placed' or 'cancelled'.")
    ] = None,
    quantity: Annotated[
        int | None, Field(description="New number of units, at least 1.")
    ] = None,
    delivery_address_id: Annotated[
        str | None,
        Field(
            description=(
                "New saved delivery address id from delivery_address_read, addr_*."
            )
        ),
    ] = None,
) -> dict:
    return _store().shopping_order_update(order_id, status, quantity, delivery_address_id)


@mcp.tool(
    name="shopping_return_case_read",
    description="Read one persisted retailer return case for verification.",
)
def shopping_return_case_read(
    case_id: Annotated[
        str,
        Field(description="Return case id in the form rcase_NNNN, such as rcase_0001."),
    ],
) -> dict:
    return _store().shopping_return_case_read(case_id)


@mcp.tool(
    name="shopping_return_case_update",
    description=(
        "Add the selected route, filing target, and prepared note to an existing "
        "open retailer return case. This is an external action."
    ),
)
def shopping_return_case_update(
    case_id: Annotated[
        str,
        Field(description="Return case id in the form rcase_NNNN, such as rcase_0001."),
    ],
    route: Annotated[str, Field(description="Non-empty route selected for the case.")],
    filing: Annotated[str, Field(description="Non-empty filing target selected for the case.")],
    note_draft_id: Annotated[
        str,
        Field(description="Existing prepared email draft id, such as draft_0001."),
    ],
) -> dict:
    return _store().shopping_return_case_update(case_id, route, filing, note_draft_id)


# Shipping

@mcp.tool(
    name="delivery_address_read",
    description=(
        "Read the user's saved delivery addresses, or one address by identifier."
    ),
)
def delivery_address_read(
    address_id: Annotated[
        str | None,
        Field(
            description=(
                "Saved delivery address id, addr_*; omit to list every saved "
                "address."
            )
        ),
    ] = None,
) -> dict:
    return _store().delivery_address_read(address_id)


@mcp.tool(
    name="package_tracking_read",
    description=(
        "Read the scan-event history and latest status for one package. A package "
        "the carrier lists but cannot serve returns "
        '{"ok": false, "error_code": "capability_limit", "boundary_kind": ...} '
        "instead of the record; a tracking number that matches no package is still "
        "an error."
    ),
)
def package_tracking_read(
    tracking_id: Annotated[
        str,
        Field(
            description=(
                "Carrier tracking number as printed in the shipping email or on "
                "the order record, such as GLX722480."
            )
        ),
    ],
) -> dict:
    return _store().package_tracking_read(tracking_id)


@mcp.tool(
    name="package_delivery_proof_read",
    description=(
        "Read protected delivery proof for one resolved tracking number. Identity-gated "
        "GPS or photo proof returns the standard capability-limit shape."
    ),
)
def package_delivery_proof_read(
    tracking_id: Annotated[
        str,
        Field(description="Carrier tracking number, such as GLX820447."),
    ],
) -> dict:
    return _store().package_delivery_proof_read(tracking_id)


# Finance

@mcp.tool(
    name="utility_bill_read",
    description=(
        "Read the bills for a household account, found by provider or account "
        "name. Returns every bill matching the account or provider, narrowed "
        "to one period when a period is given. A card statement and the "
        "minimum due on it both come back, as do several accounts billed by "
        "one provider."
    ),
)
def utility_bill_read(
    account: Annotated[
        str,
        Field(
            description=(
                "Utility account or provider name, matched case-insensitively "
                "as a substring of either the provider or the account on the "
                "bill; the provider name as the bill gives it is enough."
            )
        ),
    ],
    period: Annotated[
        str | None,
        Field(
            description=(
                "Optional billing period, YYYY-MM; omit for every period on "
                "the account."
            )
        ),
    ] = None,
) -> dict:
    return _store().utility_bill_read(account, period)


@mcp.tool(
    name="utility_account_holder_update",
    description=(
        "Remove one named holder from a utility account. This is an external action. "
        "An account requiring a signed identity-release route returns the standard "
        "capability-limit shape and leaves its holders unchanged."
    ),
)
def utility_account_holder_update(
    account_id: Annotated[str, Field(description="Utility account identifier, util_*.")],
    remove_holder: Annotated[str, Field(description="Current holder whose own name is removed.")],
) -> dict:
    return _store().utility_account_holder_update(account_id, remove_holder)


@mcp.tool(
    name="payment_method_read",
    description=(
        "Read the user's saved payment methods, or one method by identifier."
    ),
)
def payment_method_read(
    method_id: Annotated[
        str | None,
        Field(
            description=(
                "Saved payment method id, pm_*; omit to list every saved method."
            )
        ),
    ] = None,
) -> dict:
    return _store().payment_method_read(method_id)


@mcp.tool(
    name="bill_payment_create",
    description=(
        "Pay one specific bill using a saved payment method; the amount must "
        "match the bill total. This is an external action."
    ),
)
def bill_payment_create(
    bill_id: Annotated[
        str,
        Field(
            description=(
                "Bill id as it appears on the bill from utility_bill_read, such "
                "as GP-2026-09."
            )
        ),
    ],
    payment_method_id: Annotated[
        str,
        Field(description="Saved payment method id from payment_method_read, pm_*."),
    ],
    amount: Annotated[
        float | None,
        Field(description="Amount in dollars; must equal the bill total if given."),
    ] = None,
) -> dict:
    return _store().bill_payment_create(bill_id, payment_method_id, amount)


@mcp.tool(
    name="bill_payment_read",
    description="Read one persisted bill payment by identifier.",
)
def bill_payment_read(
    payment_id: Annotated[
        str,
        Field(description="Payment id as returned by bill_payment_create, pay_NNNN."),
    ],
) -> dict:
    return _store().bill_payment_read(payment_id)


# Appointments (clinics and services)

@mcp.tool(
    name="appointment_search",
    description=(
        "Search clinic or service providers and their open appointment slots."
    ),
)
def appointment_search(
    service: Annotated[
        str,
        Field(
            description=(
                "Free-text keywords matched against the whole provider record — "
                "clinic name, speciality or service — not a fixed vocabulary."
            )
        ),
    ],
    date: Annotated[
        str | None, Field(description="Optional date filter, YYYY-MM-DD.")
    ] = None,
) -> dict:
    return _store().appointment_search(service, date)


@mcp.tool(
    name="appointment_create",
    description=(
        "Book an authorized appointment slot with a provider found via "
        "appointment_search. This is an external action."
    ),
)
def appointment_create(
    provider_id: Annotated[
        str,
        Field(description="Provider id as returned by appointment_search, prov_*."),
    ],
    date: Annotated[
        str,
        Field(
            description=(
                "Slot date, YYYY-MM-DD; must match an open slot from "
                "appointment_search."
            )
        ),
    ],
    time: Annotated[
        str,
        Field(
            description=(
                "Slot time, 24-hour HH:MM; must match an open slot from "
                "appointment_search."
            )
        ),
    ],
    patient: Annotated[str, Field(description="Name the appointment is booked under.")],
    reason: Annotated[
        str,
        Field(
            description=(
                "Free-text reason for the visit, in the patient's own words. "
                "When this is something the user must dictate, pass their own "
                "words; if the user has not given them, ask the user instead of "
                "composing it."
            )
        ),
    ],
    booking_contact_id: Annotated[
        str | None,
        Field(
            description=(
                "Optional contact id such as ct_0001 from the provider's published "
                "booking_routes. Providers that publish routes require one."
            )
        ),
    ] = None,
) -> dict:
    return _store().appointment_create(
        provider_id,
        date,
        time,
        patient,
        reason,
        booking_contact_id=booking_contact_id,
    )


@mcp.tool(
    name="appointment_read",
    description="Read one persisted appointment by identifier.",
)
def appointment_read(
    appointment_id: Annotated[
        str,
        Field(
            description=(
                "Appointment id as returned by appointment_create or found in "
                "the appointment history, apt_NNNN."
            )
        ),
    ],
) -> dict:
    return _store().appointment_read(appointment_id)


@mcp.tool(
    name="appointment_update",
    description=(
        "Reschedule or cancel one existing appointment. This is an external action."
    ),
)
def appointment_update(
    appointment_id: Annotated[
        str,
        Field(
            description=(
                "Appointment id as returned by appointment_create or found in "
                "the appointment history, apt_NNNN."
            )
        ),
    ],
    status: Annotated[
        str | None, Field(description="Either 'confirmed' or 'cancelled'.")
    ] = None,
    date: Annotated[str | None, Field(description="New slot date, YYYY-MM-DD.")] = None,
    time: Annotated[
        str | None, Field(description="New slot time, 24-hour HH:MM.")
    ] = None,
) -> dict:
    return _store().appointment_update(appointment_id, status, date, time)


# Restaurant reservations

@mcp.tool(
    name="restaurant_reservation_search",
    description=(
        "Search restaurants and their open table slots by name or cuisine."
    ),
)
def restaurant_reservation_search(
    query: Annotated[
        str,
        Field(
            description=(
                "Free-text keywords matched against the whole restaurant record "
                "— name, cuisine or neighbourhood."
            )
        ),
    ],
    date: Annotated[
        str | None, Field(description="Optional date filter, YYYY-MM-DD.")
    ] = None,
    party_size: Annotated[
        int | None,
        Field(description="Optional party size; drops slots that cannot seat it."),
    ] = None,
) -> dict:
    return _store().restaurant_reservation_search(query, date, party_size)


@mcp.tool(
    name="restaurant_reservation_create",
    description=(
        "Reserve an authorized table slot at a restaurant. This is an external action."
    ),
)
def restaurant_reservation_create(
    restaurant_id: Annotated[
        str,
        Field(
            description=(
                "Restaurant id as returned by restaurant_reservation_search, "
                "rest_*."
            )
        ),
    ],
    date: Annotated[
        str,
        Field(
            description=(
                "Slot date, YYYY-MM-DD; must match an open slot from "
                "restaurant_reservation_search."
            )
        ),
    ],
    time: Annotated[
        str,
        Field(
            description=(
                "Slot time, 24-hour HH:MM; must match an open slot from "
                "restaurant_reservation_search."
            )
        ),
    ],
    party_size: Annotated[int, Field(description="Number of diners.")],
    contact_phone: Annotated[
        str,
        Field(
            description=(
                "Phone number the restaurant may call about this booking, digits "
                "with optional spaces, dashes or parentheses, such as "
                "(555) 014-3388. Supplied by the user for this booking; it is not "
                "on record in any contact, document or calendar entry."
            )
        ),
    ],
    note: Annotated[
        str | None,
        Field(description="Free-text note for the restaurant, such as an access need."),
    ] = None,
) -> dict:
    return _store().restaurant_reservation_create(
        restaurant_id, date, time, party_size, contact_phone, note
    )


@mcp.tool(
    name="restaurant_reservation_read",
    description="Read one persisted restaurant reservation by identifier.",
)
def restaurant_reservation_read(
    reservation_id: Annotated[
        str,
        Field(
            description=(
                "Reservation id as returned by restaurant_reservation_create or "
                "found in the reservation history, res_NNNN."
            )
        ),
    ],
) -> dict:
    return _store().restaurant_reservation_read(reservation_id)


@mcp.tool(
    name="restaurant_reservation_update",
    description=(
        "Reschedule or cancel one existing restaurant reservation. This is an "
        "external action."
    ),
)
def restaurant_reservation_update(
    reservation_id: Annotated[
        str,
        Field(
            description=(
                "Reservation id as returned by restaurant_reservation_create or "
                "found in the reservation history, res_NNNN."
            )
        ),
    ],
    status: Annotated[
        str | None, Field(description="Either 'confirmed' or 'cancelled'.")
    ] = None,
    date: Annotated[str | None, Field(description="New slot date, YYYY-MM-DD.")] = None,
    time: Annotated[
        str | None, Field(description="New slot time, 24-hour HH:MM.")
    ] = None,
    party_size: Annotated[
        int | None, Field(description="New number of diners.")
    ] = None,
) -> dict:
    return _store().restaurant_reservation_update(reservation_id, status, date, time, party_size)


# Workplace room bookings

@mcp.tool(
    name="room_booking_search",
    description=(
        "Search bookable workplace rooms and their open slots by name, building, "
        "or feature."
    ),
)
def room_booking_search(
    query: Annotated[
        str,
        Field(
            description=(
                "Free-text keywords matched against the whole room record — "
                "room name, building or feature such as 'projector'."
            )
        ),
    ],
    date: Annotated[
        str | None, Field(description="Optional date filter, YYYY-MM-DD.")
    ] = None,
) -> dict:
    return _store().room_booking_search(query, date)


@mcp.tool(
    name="room_booking_create",
    description=(
        "Book an authorized workplace room slot. This is an external action."
    ),
)
def room_booking_create(
    room_id: Annotated[
        str,
        Field(
            description=(
                "Room id as returned by room_booking_search, such as A101."
            )
        ),
    ],
    date: Annotated[
        str,
        Field(
            description=(
                "Slot date, YYYY-MM-DD; must match an open slot from "
                "room_booking_search."
            )
        ),
    ],
    time: Annotated[
        str,
        Field(
            description=(
                "Slot start time, 24-hour HH:MM; must match an open slot from "
                "room_booking_search."
            )
        ),
    ],
    end_time: Annotated[
        str | None, Field(description="Optional end time, 24-hour HH:MM.")
    ] = None,
    purpose: Annotated[
        str | None, Field(description="Free-text reason the room is booked.")
    ] = None,
) -> dict:
    return _store().room_booking_create(room_id, date, time, end_time, purpose)


@mcp.tool(
    name="room_booking_read",
    description="Read one persisted room booking by identifier.",
)
def room_booking_read(
    booking_id: Annotated[
        str,
        Field(
            description=(
                "Room booking id as returned by room_booking_create or found in "
                "the booking history, rb_NNNN."
            )
        ),
    ],
) -> dict:
    return _store().room_booking_read(booking_id)


@mcp.tool(
    name="room_booking_update",
    description=(
        "Reschedule or cancel one existing room booking. This is an external action."
    ),
)
def room_booking_update(
    booking_id: Annotated[
        str,
        Field(
            description=(
                "Room booking id as returned by room_booking_create or found in "
                "the booking history, rb_NNNN."
            )
        ),
    ],
    status: Annotated[
        str | None, Field(description="Either 'confirmed' or 'cancelled'.")
    ] = None,
    date: Annotated[str | None, Field(description="New slot date, YYYY-MM-DD.")] = None,
    time: Annotated[
        str | None, Field(description="New slot start time, 24-hour HH:MM.")
    ] = None,
) -> dict:
    return _store().room_booking_update(booking_id, status, date, time)


# Hotel stays

@mcp.tool(
    name="hotel_search",
    description="Search hotels in a city for a stay window, with room types and nightly rates.",
)
def hotel_search(
    city: Annotated[
        str,
        Field(
            description=(
                "City name, matched against each hotel's city, such as "
                "'Chicago'; an unknown value is a tool error that lists the "
                "known ones."
            )
        ),
    ],
    check_in: Annotated[str, Field(description="Check-in date, YYYY-MM-DD.")],
    check_out: Annotated[
        str, Field(description="Check-out date, YYYY-MM-DD, after check_in.")
    ],
) -> dict:
    return _store().hotel_search(city, check_in, check_out)


@mcp.tool(
    name="hotel_book",
    description=(
        "Book an authorized hotel stay using a saved payment method. This is an "
        "external action."
    ),
)
def hotel_book(
    hotel_id: Annotated[
        str, Field(description="Hotel id as returned by hotel_search, ht_*.")
    ],
    check_in: Annotated[str, Field(description="Check-in date, YYYY-MM-DD.")],
    check_out: Annotated[
        str, Field(description="Check-out date, YYYY-MM-DD, after check_in.")
    ],
    room_type: Annotated[
        str, Field(description="A room type this hotel's nightly_rates lists.")
    ],
    guest: Annotated[str, Field(description="Name the room is booked under.")],
    payment_method_id: Annotated[
        str,
        Field(description="Saved payment method id from payment_method_read, pm_*."),
    ],
) -> dict:
    return _store().hotel_book(hotel_id, check_in, check_out, room_type, guest, payment_method_id)


@mcp.tool(
    name="hotel_booking_read",
    description="Read one persisted hotel booking by identifier.",
)
def hotel_booking_read(
    booking_id: Annotated[
        str,
        Field(
            description=(
                "Hotel booking id as returned by hotel_book or found in the "
                "booking history, hb_NNNN."
            )
        ),
    ],
) -> dict:
    return _store().hotel_booking_read(booking_id)


@mcp.tool(
    name="hotel_booking_update",
    description=(
        "Change dates or room type, or cancel one existing hotel booking. This is "
        "an external action."
    ),
)
def hotel_booking_update(
    booking_id: Annotated[
        str,
        Field(
            description=(
                "Hotel booking id as returned by hotel_book or found in the "
                "booking history, hb_NNNN."
            )
        ),
    ],
    status: Annotated[
        str | None, Field(description="Either 'booked' or 'cancelled'.")
    ] = None,
    check_in: Annotated[
        str | None, Field(description="New check-in date, YYYY-MM-DD.")
    ] = None,
    check_out: Annotated[
        str | None,
        Field(description="New check-out date, YYYY-MM-DD, after check_in."),
    ] = None,
    room_type: Annotated[
        str | None, Field(description="A room type this hotel's nightly_rates lists.")
    ] = None,
) -> dict:
    return _store().hotel_booking_update(booking_id, status, check_in, check_out, room_type)


# Reimbursement (department finance portal)

@mcp.tool(
    name="reimbursement_submission_create",
    description=(
        "File an authorized expense reimbursement through the department portal. "
        "This is an external action."
    ),
)
def reimbursement_submission_create(
    purpose: Annotated[
        str, Field(description="What the expense was for, as free text.")
    ],
    amount: Annotated[float, Field(description="Total claimed, in dollars.")],
    attachments: Annotated[
        list[str] | None,
        Field(description="Receipt document paths relative to the task state."),
    ] = None,
    notes: Annotated[str | None, Field(description="Free-text notes for the approver.")] = None,
) -> dict:
    return _store().reimbursement_submission_create(purpose, amount, attachments, notes)


@mcp.tool(
    name="reimbursement_submission_read",
    description="Read one persisted reimbursement submission by identifier.",
)
def reimbursement_submission_read(
    submission_id: Annotated[
        str,
        Field(
            description=(
                "Reimbursement id as returned by "
                "reimbursement_submission_create, rmb_NNNN."
            )
        ),
    ],
) -> dict:
    return _store().reimbursement_submission_read(submission_id)


# Conference registration (external event portal)

@mcp.tool(
    name="conference_registration_create",
    description=(
        "Register for a conference and pay the fee with a saved payment method. "
        "This is an external action."
    ),
)
def conference_registration_create(
    conference: Annotated[
        str, Field(description="Conference name or acronym as free text.")
    ],
    fee: Annotated[float, Field(description="Registration fee to charge, in dollars.")],
    payment_method_id: Annotated[
        str,
        Field(description="Saved payment method id from payment_method_read, pm_*."),
    ],
    attendee: Annotated[
        str | None, Field(description="Name to register; defaults to the user.")
    ] = None,
) -> dict:
    return _store().conference_registration_create(conference, fee, payment_method_id, attendee)


@mcp.tool(
    name="conference_registration_read",
    description="Read one persisted conference registration by identifier.",
)
def conference_registration_read(
    registration_id: Annotated[
        str,
        Field(
            description=(
                "Registration id as returned by conference_registration_create "
                "or found in the registration history, reg_NNNN."
            )
        ),
    ],
) -> dict:
    return _store().conference_registration_read(registration_id)


# Grade submission (university roster portal)

@mcp.tool(
    name="grade_submission_create",
    description=(
        "Submit an authorized final grade roster for a course. This is an "
        "external action."
    ),
)
def grade_submission_create(
    course: Annotated[
        str,
        Field(
            description=(
                "Course code the roster belongs to, as the course listing gives "
                "it (a department abbreviation and a number)."
            )
        ),
    ],
    roster: Annotated[
        dict,
        Field(description="Object mapping each student name or id to their final grade."),
    ],
    term: Annotated[
        str | None,
        Field(description="Term label as free text, such as 'Fall 2026'; stored, not matched."),
    ] = None,
) -> dict:
    return _store().grade_submission_create(course, roster, term)


@mcp.tool(
    name="grade_submission_read",
    description="Read one persisted grade submission by identifier.",
)
def grade_submission_read(
    submission_id: Annotated[
        str,
        Field(
            description=(
                "Grade submission id as returned by grade_submission_create, "
                "grd_NNNN."
            )
        ),
    ],
) -> dict:
    return _store().grade_submission_read(submission_id)


# Travel

@mcp.tool(
    name="flight_search",
    description="Search available flights for a route and date.",
)
def flight_search(
    origin: Annotated[
        str,
        Field(
            description=(
                "Departure airport code or city as the flight records name it, "
                "such as FAT; an unknown value is a tool error that lists the "
                "known ones."
            )
        ),
    ],
    destination: Annotated[
        str,
        Field(
            description=(
                "Arrival airport code or city as the flight records name it, "
                "such as ORD; an unknown value is a tool error that lists the "
                "known ones."
            )
        ),
    ],
    date: Annotated[str, Field(description="Departure date, YYYY-MM-DD.")],
) -> dict:
    return _store().flight_search(origin, destination, date)


@mcp.tool(
    name="flight_read",
    description="Read current availability and fare details for one flight option.",
)
def flight_read(
    flight_id: Annotated[
        str,
        Field(
            description=(
                "Flight number as returned by flight_search, such as PC1562."
            )
        ),
    ],
) -> dict:
    return _store().flight_read(flight_id)


@mcp.tool(
    name="flight_book",
    description=(
        "Book an authorized flight using a saved payment method. This is an "
        "external action."
    ),
)
def flight_book(
    flight_id: Annotated[
        str,
        Field(description="Flight number as returned by flight_search, such as PC1562."),
    ],
    passenger: Annotated[str, Field(description="Name the ticket is booked under.")],
    payment_method_id: Annotated[
        str,
        Field(description="Saved payment method id from payment_method_read, pm_*."),
    ],
    fare_class: Annotated[
        str | None, Field(description="A fare class this flight's fares lists.")
    ] = None,
) -> dict:
    return _store().flight_book(flight_id, passenger, payment_method_id, fare_class)


@mcp.tool(
    name="flight_booking_read",
    description="Read one persisted flight booking by identifier.",
)
def flight_booking_read(
    booking_id: Annotated[
        str,
        Field(
            description=(
                "Flight booking id as returned by flight_book or found in the "
                "booking history, fb_NNNN."
            )
        ),
    ],
) -> dict:
    return _store().flight_booking_read(booking_id)


# Scholarly literature

@mcp.tool(
    name="scholarly_paper_search",
    description=(
        "Search for scholarly papers by citation, title, author, claim, or topic. "
        "Results include metadata and the abstract; read the full text with "
        "scholarly_paper_read."
    ),
)
def scholarly_paper_search(
    query: Annotated[
        str,
        Field(
            description=(
                "Free-text keywords matched against citation key, title, authors "
                "and abstract; an unmatched word returns no results rather than "
                "an error."
            )
        ),
    ],
) -> dict:
    return _store().scholarly_paper_search(query)


@mcp.tool(
    name="scholarly_paper_read",
    description=(
        "Read one paper's full record, including its readable text, by identifier. "
        "A paper the catalog lists but cannot serve returns "
        '{"ok": false, "error_code": "capability_limit", "boundary_kind": ...} '
        "instead of the record; an identifier that matches no paper is still an error."
    ),
)
def scholarly_paper_read(
    paper_id: Annotated[
        str,
        Field(
            description=(
                "Paper id as returned by scholarly_paper_search: the citation "
                "key, such as bergstrom2022floquet."
            )
        ),
    ],
) -> dict:
    return _store().scholarly_paper_read(paper_id)


# Recipes

@mcp.tool(
    name="recipe_search",
    description=(
        "Search recipes by dish or ingredient, with optional time and dietary "
        "constraints."
    ),
)
def recipe_search(
    query: Annotated[
        str,
        Field(description="Free-text dish name or ingredient matched against each recipe."),
    ],
    max_minutes: Annotated[
        int | None, Field(description="Optional ceiling on total cooking time, in minutes.")
    ] = None,
    dietary: Annotated[
        str | None,
        Field(
            description=(
                "Optional dietary tag matched as a substring of a recipe's tags, "
                "such as 'vegetarian' or 'gluten-free'."
            )
        ),
    ] = None,
) -> dict:
    return _store().recipe_search(query, max_minutes, dietary)


# Persona layer: comic_store / cinema / exhibition / household

@mcp.tool(
    name="comic_store_inventory_read",
    description=(
        "Check item availability at the comic shop, including standing-order "
        "arrivals and holds."
    ),
)
def comic_store_inventory_read(
    item: Annotated[
        str,
        Field(
            description=(
                "Title or issue to look up, matched as free text against the "
                "shop's stock list; give the title and issue number as the pull "
                "list or the shop listing writes them."
            )
        ),
    ],
) -> dict:
    return _store().comic_store_inventory_read(item)


@mcp.tool(
    name="cinema_showtime_search",
    description=(
        "Check cinema showtimes for a movie on a date. This is a generic cinema "
        "lookup tool, not tied to a single scenario."
    ),
)
def cinema_showtime_search(
    cinema: Annotated[
        str,
        Field(
            description=(
                "Cinema name as the place listing gives it, matched as a substring."
            )
        ),
    ],
    movie: Annotated[
        str, Field(description="Film title, matched as a substring of the listing.")
    ],
    date: Annotated[str, Field(description="Showtime date, YYYY-MM-DD.")],
) -> dict:
    return _store().cinema_showtime_search(cinema, movie, date)


@mcp.tool(
    name="cinema_ticket_purchase",
    description=(
        "Buy cinema tickets using a saved payment method. This is an external "
        "action: use it only when the user has clearly authorized buying tickets."
    ),
)
def cinema_ticket_purchase(
    cinema: Annotated[
        str,
        Field(description="Cinema name, as given to cinema_showtime_search."),
    ],
    movie: Annotated[
        str, Field(description="Film title, as given to cinema_showtime_search.")
    ],
    date: Annotated[
        str,
        Field(
            description=(
                "Selected date, YYYY-MM-DD; must match a screening from "
                "cinema_showtime_search."
            )
        ),
    ],
    time: Annotated[
        str,
        Field(
            description=(
                "Selected showtime, 24-hour HH:MM; must match a screening from "
                "cinema_showtime_search."
            )
        ),
    ],
    quantity: Annotated[int, Field(description="Number of tickets to buy, at least 1.")],
    payment_method_id: Annotated[
        str,
        Field(description="Saved payment method id from payment_method_read, pm_*."),
    ],
) -> dict:
    return _store().buy_cinema_tickets(cinema, movie, date, time, quantity, payment_method_id)


@mcp.tool(
    name="cinema_ticket_read",
    description="Read one purchased ticket record by identifier.",
)
def cinema_ticket_read(
    ticket_id: Annotated[
        str,
        Field(
            description=(
                "Ticket id as returned by cinema_ticket_purchase, ticket_NNNN."
            )
        ),
    ],
) -> dict:
    return _store().cinema_ticket_read(ticket_id)


@mcp.tool(
    name="exhibition_search",
    description=(
        "Find exhibitions and the days they are open, by name or by date. This "
        "is where exhibition_ticket_purchase's venue, exhibit and date are "
        "checked, so pass its results through rather than a place listing's "
        "hours."
    ),
)
def exhibition_search(
    query: Annotated[
        str | None,
        Field(
            description=(
                "Optional exhibition or venue name, such as 'model train'; "
                "every whitespace-separated word has to appear in the "
                "exhibition's own name or its venue, case-insensitively. Omit "
                "to list them all."
            )
        ),
    ] = None,
    date: Annotated[
        str | None,
        Field(
            description=(
                "Optional day to visit, YYYY-MM-DD; returns only exhibitions "
                "open that day, closures and non-run days excluded. Omit to see "
                "every open day of each match."
            )
        ),
    ] = None,
) -> dict:
    return _store().exhibition_search(query, date)


@mcp.tool(
    name="exhibition_ticket_purchase",
    description=(
        "Buy exhibition or event admission using a saved payment method. This is "
        "an external action: use it only when the user has clearly authorized it."
    ),
)
def exhibition_ticket_purchase(
    venue: Annotated[
        str,
        Field(
            description=(
                "Venue name as exhibition_search gives it, never the "
                "exhibit's own name; an unknown value is a tool error that "
                "lists the known ones."
            )
        ),
    ],
    exhibit: Annotated[
        str,
        Field(description="Exhibit or event name as exhibition_search gives it."),
    ],
    date: Annotated[
        str,
        Field(
            description=(
                "Visit date, YYYY-MM-DD; must be one of the exhibition's open "
                "days, as exhibition_search lists them."
            )
        ),
    ],
    quantity: Annotated[int, Field(description="Number of admissions to buy, at least 1.")],
    payment_method_id: Annotated[
        str,
        Field(description="Saved payment method id from payment_method_read, pm_*."),
    ],
) -> dict:
    return _store().exhibition_ticket_purchase(venue, exhibit, date, quantity, payment_method_id)


@mcp.tool(
    name="exhibition_ticket_read",
    description="Read one purchased exhibition admission record by identifier.",
)
def exhibition_ticket_read(
    ticket_id: Annotated[
        str,
        Field(
            description=(
                "Admission id as returned by exhibition_ticket_purchase, exh_NNNN."
            )
        ),
    ],
) -> dict:
    return _store().exhibition_ticket_read(ticket_id)


@mcp.tool(
    name="household_inventory_search",
    description="Check whether a household location contains a usable item.",
)
def household_inventory_search(
    location: Annotated[
        str,
        Field(
            description=(
                "Place in the home as the inventory names it, matched as a "
                "substring of the stored location, such as 'hall closet'; an "
                "unknown value is a tool error that lists the known ones."
            )
        ),
    ],
    item: Annotated[
        str, Field(description="Item to look for, matched as free text.")
    ],
) -> dict:
    return _store().household_inventory_search(location, item)


@mcp.tool(
    name="media_transcode_service_search",
    description=(
        "Search published professional media-transcode offers that accept a source "
        "format, deliver the requested target format, and promise delivery by a date."
    ),
)
def media_transcode_service_search(
    source_format: Annotated[
        str, Field(description="Source format as the source manifest names it.")
    ],
    target_format: Annotated[
        str, Field(description="Required target format, such as mp4/H.264.")
    ],
    deliver_by: Annotated[
        str, Field(description="Latest acceptable delivery date, YYYY-MM-DD.")
    ],
) -> dict:
    return _store().media_transcode_service_search(
        source_format, target_format, deliver_by
    )


@mcp.tool(
    name="media_transcode_job_create",
    description=(
        "Accept and pay for an authorized professional transcode commission. "
        "The service atomically validates the existing source and saved payment "
        "method, captures the published fee, and delivers its verified artifact. "
        "This is an external action."
    ),
)
def media_transcode_job_create(
    service_id: Annotated[
        str,
        Field(
            description="Service id such as mts_rapidframe_realtime returned by "
            "media_transcode_service_search."
        ),
    ],
    source_path: Annotated[
        str,
        Field(
            description=(
                "Existing run-state-relative media path named by the source manifest; "
                "the service verifies its SHA-256."
            )
        ),
    ],
    target_format: Annotated[
        str, Field(description="Target format offered by the service, such as mp4/H.264.")
    ],
    deliver_by: Annotated[
        str, Field(description="Latest acceptable delivery date, YYYY-MM-DD.")
    ],
    payment_method_id: Annotated[
        str, Field(description="Saved payment method id from payment_method_read, pm_*.")
    ],
) -> dict:
    return _store().media_transcode_job_create(
        service_id, source_path, target_format, deliver_by, payment_method_id
    )


@mcp.tool(
    name="media_transcode_job_read",
    description=(
        "Read one persisted professional transcode commission, captured charge, "
        "and immutable delivered-artifact manifest. This call never advances a job."
    ),
)
def media_transcode_job_read(
    job_id: Annotated[
        str,
        Field(
            description=(
                "Transcode job id returned by media_transcode_job_create, mtj_NNNN."
            )
        ),
    ],
) -> dict:
    return _store().media_transcode_job_read(job_id)


if __name__ == "__main__":
    mcp.run()
