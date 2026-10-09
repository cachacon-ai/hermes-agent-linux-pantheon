"""Internal metadata attached to durable conversation messages."""

from __future__ import annotations

from time import time as wall_time
from typing import Any, MutableMapping, Optional, TypeVar


# These fields describe Hermes' durable record, not provider-visible message
# content. They must not influence context-pressure decisions.
PERSISTENCE_ONLY_MESSAGE_FIELDS = frozenset({"timestamp", "display_metadata", "display_kind"})

# Stored in the existing display sidecar, which never reaches model context.
# Physical row ids change during compaction; this original committed row id
# identifies the same authored assistant message across its carried clones.
REPLY_SOURCE_ROW_ID_KEY = "_reply_source_row_id"


def valid_reply_row_id(value: Any) -> bool:
    """Whether an identity survives JSON transport to JavaScript exactly."""
    return type(value) is int and 0 < value <= 2**53 - 1

_Message = TypeVar("_Message", bound=MutableMapping[str, Any])


def stamp_message_timestamp(
    message: _Message,
    *,
    timestamp: Optional[float] = None,
) -> _Message:
    """Attach a creation timestamp without replacing source-provided time.

    Gateway adapters can supply the platform event time. All other callers use
    the local wall clock at the point the message enters the live transcript.
    Returning the same mapping keeps the helper convenient at append sites.
    """
    if message.get("timestamp") is None:
        message["timestamp"] = wall_time() if timestamp is None else timestamp
    return message


def append_message(
    messages: list[Any],
    message: _Message,
    *,
    timestamp: Optional[float] = None,
) -> _Message:
    """Stamp and append one live transcript message."""
    stamp_message_timestamp(message, timestamp=timestamp)
    messages.append(message)
    return message
