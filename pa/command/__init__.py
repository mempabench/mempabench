"""Slash command routing and built-in handlers."""

from pa.command.builtin import register_builtin_commands
from pa.command.router import CommandContext, CommandRouter

__all__ = ["CommandContext", "CommandRouter", "register_builtin_commands"]
