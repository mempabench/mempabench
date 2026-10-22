"""Message bus module for decoupled channel-agent communication."""

from pa.bus.events import InboundMessage, OutboundMessage
from pa.bus.queue import MessageBus

__all__ = ["MessageBus", "InboundMessage", "OutboundMessage"]
