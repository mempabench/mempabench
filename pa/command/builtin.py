"""Built-in slash command handlers."""

from __future__ import annotations

import asyncio
import os
import sys

from memory.event_writer import consolidation_records, records_owed_to_files
from pa import __version__
from pa.bus.events import OutboundMessage
from pa.command.router import CommandContext, CommandRouter
from pa.utils.helpers import build_status_content


async def cmd_stop(ctx: CommandContext) -> OutboundMessage:
    """Cancel all active tasks for the session."""
    loop = ctx.loop
    msg = ctx.msg
    tasks = loop._active_tasks.pop(msg.session_key, [])
    cancelled = sum(1 for t in tasks if not t.done() and t.cancel())
    for t in tasks:
        try:
            await t
        except (asyncio.CancelledError, Exception):
            pass
    content = f"Stopped {cancelled} task(s)." if cancelled else "No active task to stop."
    return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content=content)


async def cmd_restart(ctx: CommandContext) -> OutboundMessage:
    """Restart the process in-place via os.execv."""
    msg = ctx.msg

    async def _do_restart():
        await asyncio.sleep(1)
        os.execv(sys.executable, [sys.executable, "-m", "nanobot"] + sys.argv[1:])

    asyncio.create_task(_do_restart())
    return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content="Restarting...")


async def cmd_status(ctx: CommandContext) -> OutboundMessage:
    """Build an outbound status message for a session."""
    loop = ctx.loop
    session = ctx.session or loop.sessions.get(ctx.key)
    ctx_est = 0
    try:
        ctx_est, _ = loop.memory_consolidator.estimate_session_prompt_tokens(session)
    except Exception:
        pass
    if ctx_est <= 0:
        ctx_est = loop._last_usage.get("prompt_tokens", 0)
    return OutboundMessage(
        channel=ctx.msg.channel,
        chat_id=ctx.msg.chat_id,
        content=build_status_content(
            version=__version__, model=loop.pa_model,
            start_time=loop._start_time, last_usage=loop._last_usage,
            context_window_tokens=loop.pa_context_window_tokens,
            session_msg_count=len(session.get_history(max_messages=0)),
            context_tokens_estimate=ctx_est,
        ),
        metadata={"render_as": "text"},
    )


async def cmd_new(ctx: CommandContext) -> OutboundMessage:
    """Start a fresh session, archiving what the journal still owes first.

    The order matters and used to be the other way round: the session was
    cleared and only then archived, so a consolidation that refused the
    contract took the conversation with it. Archive first, clear only on
    success, and the command is retriable. The source is the same turn journal
    mid-session compaction and the pinned flush read -- this was
    the one entry point that archived the raw message tail with no projection
    at all.
    """
    loop = ctx.loop
    session = ctx.session or loop.sessions.get(ctx.key)
    journal = loop.turn_journal(session.key)
    owed = records_owed_to_files(journal.records())
    if owed and loop.native_file_memory_enabled:
        records = consolidation_records(owed)
        loop.memory_consolidator.preflight_archive_chunk(records)
        await loop.memory_consolidator.archive_messages_to_files(records)
        journal.mark_file_consolidated([record.turn_id for record in owed])
    session.clear()
    loop.sessions.save(session)
    loop.sessions.invalidate(session.key)
    # A cleared session restarts its spans at 0, so the journal needs a new
    # namespace or the next turn's `[0, n)` collides with the old session's.
    journal.open_boundary()
    return OutboundMessage(
        channel=ctx.msg.channel, chat_id=ctx.msg.chat_id,
        content="New session started.",
    )


async def cmd_help(ctx: CommandContext) -> OutboundMessage:
    """Return available slash commands."""
    lines = [
        "🐈 nanobot commands:",
        "/new — Start a new conversation",
        "/stop — Stop the current task",
        "/restart — Restart the bot",
        "/status — Show bot status",
        "/help — Show available commands",
    ]
    return OutboundMessage(
        channel=ctx.msg.channel,
        chat_id=ctx.msg.chat_id,
        content="\n".join(lines),
        metadata={"render_as": "text"},
    )


def register_builtin_commands(router: CommandRouter) -> None:
    """Register the default set of slash commands."""
    router.priority("/stop", cmd_stop)
    router.priority("/restart", cmd_restart)
    router.priority("/status", cmd_status)
    router.exact("/new", cmd_new)
    router.exact("/status", cmd_status)
    router.exact("/help", cmd_help)
