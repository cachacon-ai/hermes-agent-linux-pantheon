"""Relay must propagate connector ``content_rendered`` onto ``SendResult``.

#82656 / stream-finalization review: an ACKed-but-unapplied send/edit from the
connector used to be reconstructed as ``content_rendered=True`` (the
``SendResult`` default), so stream consumers advanced delivery bookkeeping and
the gateway suppressed its corrective final. These tests drive the real
``RelayAdapter`` send / send_for_platform / edit_message constructors.
"""

from __future__ import annotations

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.relay.adapter import RelayAdapter, _content_rendered_from_connector
from gateway.relay.descriptor import CONTRACT_VERSION, CapabilityDescriptor


def make_desc(**kw) -> CapabilityDescriptor:
    base = dict(
        contract_version=CONTRACT_VERSION,
        platform="telegram",
        label="Telegram",
        max_message_length=4096,
        supports_draft_streaming=False,
        supports_edit=True,
        supports_threads=True,
        markdown_dialect="markdown_v2",
        len_unit="utf16",
        emoji="",
        platform_hint="",
        pii_safe=False,
        supported_ops=("send", "edit"),
    )
    base.update(kw)
    return CapabilityDescriptor(**base)


class _ScriptedTransport:
    """Returns a canned outbound result and records the action."""

    def __init__(self, result: dict):
        self._result = result
        self.sent = None
        self._identities = []

    def set_inbound_handler(self, h):
        self._h = h

    async def send_outbound(self, action, *, platform=None):
        self.sent = action
        return dict(self._result)


def _adapter(result: dict) -> RelayAdapter:
    return RelayAdapter(
        PlatformConfig(),
        make_desc(),
        transport=_ScriptedTransport(result),
    )


@pytest.mark.parametrize(
    "payload,expected",
    [
        ({}, True),
        ({"success": True}, True),
        ({"success": True, "content_rendered": True}, True),
        ({"success": True, "content_rendered": False}, False),
        ({"success": True, "content_rendered": 0}, False),
        ("not-a-dict", True),
        (None, True),
    ],
)
def test_content_rendered_from_connector_mapping(payload, expected):
    assert _content_rendered_from_connector(payload) is expected


@pytest.mark.asyncio
async def test_edit_message_propagates_unrendered_ack():
    adapter = _adapter(
        {
            "success": True,
            "message_id": "m-edit",
            "content_rendered": False,
        }
    )
    result = await adapter.edit_message("chat-1", "m-1", "hello")
    assert result.success is True
    assert result.content_rendered is False
    assert result.message_id == "m-edit"


@pytest.mark.asyncio
async def test_edit_message_defaults_rendered_when_connector_omits_field():
    adapter = _adapter({"success": True, "message_id": "m-edit"})
    result = await adapter.edit_message("chat-1", "m-1", "hello")
    assert result.success is True
    assert result.content_rendered is True


@pytest.mark.asyncio
async def test_send_propagates_unrendered_ack():
    adapter = _adapter(
        {
            "success": True,
            "message_id": "m-send",
            "content_rendered": False,
        }
    )
    result = await adapter.send("chat-1", "hello")
    assert result.success is True
    assert result.content_rendered is False
    assert result.message_id == "m-send"


@pytest.mark.asyncio
async def test_send_for_platform_propagates_unrendered_ack():
    transport = _ScriptedTransport(
        {
            "success": True,
            "message_id": "m-sfp",
            "content_rendered": False,
        }
    )
    transport._identities = [("telegram", "bot-1")]
    adapter = RelayAdapter(
        PlatformConfig(),
        make_desc(platform="telegram"),
        transport=transport,
    )
    result = await adapter.send_for_platform(Platform.TELEGRAM, "chat-1", "hello")
    assert result.success is True
    assert result.content_rendered is False
    assert result.message_id == "m-sfp"
