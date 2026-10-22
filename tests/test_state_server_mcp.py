import asyncio

from harness.state_runtime import initialize_state_runtime
from pa.agent.tools.mcp import connect_mcp_servers
from pa.agent.tools.registry import ToolRegistry


def test_state_server_mcp_tools_register_and_execute(tmp_path):
    async def run():
        from contextlib import AsyncExitStack

        registry = ToolRegistry()
        async with AsyncExitStack() as stack:
            await connect_mcp_servers(runtime.mcp_server_config(), registry, stack)
            assert registry.has("mcp_state_document_read")
            assert registry.has("mcp_state_email_draft_create")
            assert not registry.has("mcp_state_email_message_send")

            doc = await registry.execute(
                "mcp_state_document_read",
                {"path": "documents/intro.md"},
            )
            assert "mcp draft" in str(doc)

            draft = await registry.execute(
                "mcp_state_email_draft_create",
                {"to": "x@example.test", "subject": "Subject", "body": "Body"},
            )
            assert "draft_0001" in str(draft)

    fixture = tmp_path / "fixtures" / "user_a"
    (fixture / "documents").mkdir(parents=True)
    (fixture / "email").mkdir()
    (fixture / "documents" / "intro.md").write_text("mcp draft", encoding="utf-8")
    (fixture / "email" / "drafts.jsonl").write_text("", encoding="utf-8")
    (fixture / "email" / "sent.jsonl").write_text("", encoding="utf-8")

    runtime = initialize_state_runtime(
        user_id="user_a",
        session_id="session_02",
        run_id="run_001",
        fixture_root=tmp_path / "fixtures",
        runs_root=tmp_path / "runs",
        project_root=".",
    )

    asyncio.run(run())

def test_state_server_mcp_registers_document_update_when_enabled(tmp_path):
    async def run():
        from contextlib import AsyncExitStack

        registry = ToolRegistry()
        async with AsyncExitStack() as stack:
            await connect_mcp_servers(runtime.mcp_server_config(), registry, stack)
            assert registry.has("mcp_state_document_update")
            assert registry.has("mcp_state_document_create")
            assert registry.has("mcp_state_document_delete")

            updated = await registry.execute(
                "mcp_state_document_update",
                {
                    "path": "documents/intro.md",
                    "anchor": "mcp draft",
                    "replacement": "updated mcp draft",
                },
            )
            assert "updated" in str(updated)

            created = await registry.execute(
                "mcp_state_document_create",
                {"path": "documents/second.md", "content": "second doc"},
            )
            assert "created" in str(created)
            existing = await registry.execute(
                "mcp_state_document_create",
                {"path": "documents/second.md", "content": "clobbered"},
            )
            assert "exists" in str(existing)
            deleted = await registry.execute(
                "mcp_state_document_delete", {"path": "documents/second.md"}
            )
            assert "deleted" in str(deleted)

    fixture = tmp_path / "fixtures" / "user_a"
    (fixture / "documents").mkdir(parents=True)
    (fixture / "email").mkdir()
    (fixture / "documents" / "intro.md").write_text("mcp draft", encoding="utf-8")
    (fixture / "email" / "drafts.jsonl").write_text("", encoding="utf-8")

    runtime = initialize_state_runtime(
        user_id="user_a",
        session_id="session_03",
        run_id="run_003",
        fixture_root=tmp_path / "fixtures",
        runs_root=tmp_path / "runs",
        project_root=".",
        enabled_tools=[
            "document_read",
            "document_create",
            "document_update",
            "document_delete",
            "email_draft_create",
        ],
    )

    asyncio.run(run())
    updated_doc = runtime.state_root / "documents" / "intro.md"
    assert updated_doc.read_text(encoding="utf-8") == "updated mcp draft"


def test_state_server_mcp_registers_cinema_ticket_purchase_when_enabled(tmp_path):
    async def run():
        from contextlib import AsyncExitStack

        registry = ToolRegistry()
        async with AsyncExitStack() as stack:
            await connect_mcp_servers(runtime.mcp_server_config(), registry, stack)
            assert registry.has("mcp_state_cinema_ticket_purchase")

            bought = await registry.execute(
                "mcp_state_cinema_ticket_purchase",
                {
                    "cinema": "Glenmont Cinema",
                    "movie": "Temporal Echo",
                    "date": "2026-07-21",
                    "time": "20:00",
                    "quantity": 2,
                    "payment_method_id": "pm_visa_4821",
                },
            )
            assert "ticket_0001" in str(bought)

    fixture = tmp_path / "fixtures" / "user_a"
    (fixture / "documents").mkdir(parents=True)
    (fixture / "cinema").mkdir()
    (fixture / "cinema" / "showtimes.json").write_text(
        '[{"cinema":"Glenmont Cinema","movie":"Temporal Echo",'
        '"date":"2026-07-21","time":"20:00"}]',
        encoding="utf-8",
    )
    (fixture / "finance").mkdir()
    (fixture / "finance" / "payment_methods.json").write_text(
        '{"payment_methods":[{"method_id":"pm_visa_4821","kind":"credit_card"}]}',
        encoding="utf-8",
    )

    runtime = initialize_state_runtime(
        user_id="user_a",
        session_id="session_039",
        run_id="run_039",
        fixture_root=tmp_path / "fixtures",
        runs_root=tmp_path / "runs",
        project_root=".",
        enabled_tools=["document_read", "email_draft_create", "cinema_ticket_purchase"],
    )

    asyncio.run(run())
    tickets = runtime.state_root / "cinema" / "tickets.jsonl"
    assert "Temporal Echo" in tickets.read_text(encoding="utf-8")


def test_state_server_mcp_session_09_registers_email_message_send(tmp_path):
    async def run():
        from contextlib import AsyncExitStack

        registry = ToolRegistry()
        async with AsyncExitStack() as stack:
            await connect_mcp_servers(runtime.mcp_server_config(), registry, stack)
            assert registry.has("mcp_state_email_draft_create")
            assert registry.has("mcp_state_email_message_send")

            sent = await registry.execute(
                "mcp_state_email_message_send",
                {"to": "x@example.test", "subject": "Subject", "body": "Body"},
            )
            assert "sent_0001" in str(sent)

    fixture = tmp_path / "fixtures" / "user_a"
    (fixture / "documents").mkdir(parents=True)
    (fixture / "email").mkdir()
    (fixture / "email" / "drafts.jsonl").write_text("", encoding="utf-8")
    (fixture / "email" / "sent.jsonl").write_text("", encoding="utf-8")

    runtime = initialize_state_runtime(
        user_id="user_a",
        session_id="session_09",
        run_id="run_009",
        fixture_root=tmp_path / "fixtures",
        runs_root=tmp_path / "runs",
        project_root=".",
        enabled_tools=["document_read", "email_draft_create", "email_message_send"],
    )

    asyncio.run(run())


def test_state_server_mcp_session_14_registers_internal_write_boundary_tools(tmp_path):
    async def run():
        from contextlib import AsyncExitStack

        registry = ToolRegistry()
        async with AsyncExitStack() as stack:
            await connect_mcp_servers(runtime.mcp_server_config(), registry, stack)
            assert registry.has("mcp_state_calendar_event_create")
            assert registry.has("mcp_state_email_message_send")

            created = await registry.execute(
                "mcp_state_calendar_event_create",
                {"title": "Train exhibition visit", "date": "2026-10-18"},
            )
            assert "ev_0001" in str(created)

    fixture = tmp_path / "fixtures" / "user_a"
    (fixture / "documents").mkdir(parents=True)
    (fixture / "email").mkdir()
    (fixture / "calendar").mkdir()
    (fixture / "email" / "drafts.jsonl").write_text("", encoding="utf-8")
    (fixture / "email" / "sent.jsonl").write_text("", encoding="utf-8")

    runtime = initialize_state_runtime(
        user_id="user_a",
        session_id="session_14",
        run_id="run_014",
        fixture_root=tmp_path / "fixtures",
        runs_root=tmp_path / "runs",
        project_root=".",
        enabled_tools=["document_read", "email_draft_create", "calendar_event_create", "email_message_send"],
    )

    asyncio.run(run())


def test_state_server_mcp_route_planning_tools_register_and_execute(tmp_path):
    async def run():
        from contextlib import AsyncExitStack

        registry = ToolRegistry()
        async with AsyncExitStack() as stack:
            await connect_mcp_servers(runtime.mcp_server_config(), registry, stack)
            assert registry.has("mcp_state_email_message_search")
            assert registry.has("mcp_state_email_message_read")
            assert registry.has("mcp_state_map_route_search")

            email = await registry.execute("mcp_state_email_message_search", {"query": "train"})
            assert "Glenmont Model Train Exhibition" in str(email)

            full_email = await registry.execute("mcp_state_email_message_read", {"email_id": "email_0001"})
            assert "Venue: Glenmont Civic Exhibition Hall" in str(full_email)

            routes = await registry.execute(
                "mcp_state_map_route_search",
                {
                    "origin": "User A's apartment",
                    "stop": "Dale's Comic Center",
                    "destination": "Glenmont Civic Exhibition Hall",
                },
            )
            assert "public_transit" in str(routes)

    fixture = tmp_path / "fixtures" / "user_a"
    (fixture / "email").mkdir(parents=True)
    (fixture / "maps").mkdir()
    (fixture / "email" / "drafts.jsonl").write_text("", encoding="utf-8")
    (fixture / "email" / "inbox.jsonl").write_text(
        '{"id":"email_0001","from":"events@example.test","date":"2026-05-05",'
        '"subject":"Glenmont Model Train Exhibition","body":"Venue: Glenmont Civic Exhibition Hall."}\n',
        encoding="utf-8",
    )
    (fixture / "maps" / "route_options.json").write_text(
        "{\"origin\":\"User A's apartment\",\"stop\":\"Dale's Comic Center\","
        '"destination":"Glenmont Civic Exhibition Hall",'
        '"options":[{"mode":"public_transit","duration_minutes":52,"estimated_price_usd":4.5}]}',
        encoding="utf-8",
    )

    runtime = initialize_state_runtime(
        user_id="user_a",
        session_id="session_04",
        run_id="run_004",
        fixture_root=tmp_path / "fixtures",
        runs_root=tmp_path / "runs",
        project_root=".",
        enabled_tools=["document_read", "email_draft_create", "email_message_search", "email_message_read", "map_route_search"],
    )

    asyncio.run(run())


def test_state_server_mcp_weather_forecast_registers_and_executes(tmp_path):
    async def run():
        from contextlib import AsyncExitStack

        registry = ToolRegistry()
        async with AsyncExitStack() as stack:
            await connect_mcp_servers(runtime.mcp_server_config(), registry, stack)
            assert registry.has("mcp_state_weather_forecast_read")

            forecast = await registry.execute(
                "mcp_state_weather_forecast_read",
                {
                    "location": "Glenmont, CA",
                    "date": "2026-12-08",
                    "time_window": "18:00-22:00",
                },
            )
            assert "Bring an umbrella." in str(forecast)

    fixture = tmp_path / "fixtures" / "user_a"
    (fixture / "weather").mkdir(parents=True)
    (fixture / "weather" / "glenmont_evening_forecast.json").write_text(
        '{"forecasts":[{"location":"Glenmont, CA","date":"2026-12-08",'
        '"time_window":"18:00-22:00","forecast":{"recommendation":"Bring an umbrella."}}]}',
        encoding="utf-8",
    )

    runtime = initialize_state_runtime(
        user_id="user_a",
        session_id="evolv_05",
        run_id="run_evolv_05",
        fixture_root=tmp_path / "fixtures",
        runs_root=tmp_path / "runs",
        project_root=".",
        enabled_tools=["weather_forecast_read"],
    )

    asyncio.run(run())


def test_state_server_mcp_one_at_a_time_tools_register_and_execute(tmp_path):
    async def run():
        from contextlib import AsyncExitStack

        registry = ToolRegistry()
        async with AsyncExitStack() as stack:
            await connect_mcp_servers(runtime.mcp_server_config(), registry, stack)
            assert registry.has("mcp_state_comic_store_inventory_read")
            assert registry.has("mcp_state_cinema_showtime_search")
            assert registry.has("mcp_state_place_search")

            item = await registry.execute(
                "mcp_state_comic_store_inventory_read",
                {"item": "The Flash Archive Volume 3"},
            )
            assert "available_for_pickup" in str(item)

            showtimes = await registry.execute(
                "mcp_state_cinema_showtime_search",
                {
                    "cinema": "Glenmont Cinema",
                    "movie": "The Time Machine",
                    "date": "2026-05-07",
                },
            )
            assert "19:10" in str(showtimes)

            places = await registry.execute(
                "mcp_state_place_search",
                {
                    "location": "Glenmont Cinema",
                    "category": "dinner",
                    "limit": 2,
                },
            )
            assert "Civic Plaza Diner" in str(places)

    fixture = tmp_path / "fixtures" / "user_a"
    (fixture / "comic_store").mkdir(parents=True)
    (fixture / "cinema").mkdir()
    (fixture / "places").mkdir()
    (fixture / "email").mkdir()
    (fixture / "email" / "drafts.jsonl").write_text("", encoding="utf-8")
    (fixture / "comic_store" / "items.json").write_text(
        '[{"store":"Dale\'s Comic Center","item":"The Flash Archive Volume 3",'
        '"status":"available_for_pickup"}]',
        encoding="utf-8",
    )
    (fixture / "cinema" / "showtimes.json").write_text(
        '[{"cinema":"Glenmont Cinema","movie":"The Time Machine",'
        '"date":"2026-05-07","time":"19:10"}]',
        encoding="utf-8",
    )
    (fixture / "places" / "nearby.json").write_text(
        '[{"name":"Civic Plaza Diner","near":"Glenmont Cinema",'
        '"category":"dinner","tags":["casual"]}]',
        encoding="utf-8",
    )

    runtime = initialize_state_runtime(
        user_id="user_a",
        session_id="session_06",
        run_id="run_006",
        fixture_root=tmp_path / "fixtures",
        runs_root=tmp_path / "runs",
        project_root=".",
        enabled_tools=["document_read", "email_draft_create", "comic_store_inventory_read", "cinema_showtime_search", "place_search"],
    )

    asyncio.run(run())


def test_state_server_mcp_capability_boundary_contacts_tool(tmp_path):
    async def run():
        from contextlib import AsyncExitStack

        registry = ToolRegistry()
        async with AsyncExitStack() as stack:
            await connect_mcp_servers(runtime.mcp_server_config(), registry, stack)
            assert registry.has("mcp_state_contact_search")

            contact = await registry.execute(
                "mcp_state_contact_search",
                {"query": "quiet-entry"},
            )
            assert "accessibility@glenmontcivic.example" in str(contact)

    fixture = tmp_path / "fixtures" / "user_a"
    fixture.mkdir(parents=True)
    (fixture / "email").mkdir()
    (fixture / "email" / "drafts.jsonl").write_text("", encoding="utf-8")
    (fixture / "contacts.json").write_text(
        '{"accessibility":{"name":"Glenmont Civic Exhibition Hall Accessibility Desk",'
        '"email":"accessibility@glenmontcivic.example",'
        '"tags":["quiet-entry","accommodation"]}}',
        encoding="utf-8",
    )

    runtime = initialize_state_runtime(
        user_id="user_a",
        session_id="session_22",
        run_id="run_022",
        fixture_root=tmp_path / "fixtures",
        runs_root=tmp_path / "runs",
        project_root=".",
        enabled_tools=["document_read", "email_draft_create", "contact_search"],
    )

    asyncio.run(run())
