"""Regression coverage for unrendered ACKs on final-delivery lanes (#82656).

Covers the stream-finalization review findings that the first #82656 patch
missed:

1. Fallback final sends must require ``content_rendered`` (known False is
   failure, not ambiguous timeout).
2. Confirmed prefix + confirmed rendered tail remains complete delivery.
3. Gateway stale/transformed reconciliation must not set ``already_sent`` on
   an unrendered edit ACK.
"""

from __future__ import annotations

import importlib
import sys
import types
from types import SimpleNamespace

import pytest
import yaml

from gateway.config import Platform, PlatformConfig, StreamingConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.session import SessionSource
from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig


# ---------------------------------------------------------------------------
# Consumer-level: fallback send lanes
# ---------------------------------------------------------------------------


class _LyingSendAdapter(BasePlatformAdapter):
    """Every send ACKs but never renders; edits always fail."""

    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="***"), Platform.TELEGRAM)
        self.sends = []
        self.edits = []

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def get_chat_info(self, chat_id):
        return {}

    async def send_typing(self, chat_id, metadata=None) -> None:
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        self.sends.append(content)
        return SendResult(
            success=True,
            message_id=f"m-{len(self.sends)}",
            content_rendered=False,
        )

    async def edit_message(
        self, chat_id, message_id, content, *, finalize: bool = False, metadata=None
    ) -> SendResult:
        self.edits.append(content)
        return SendResult(success=False, error="flood control")


@pytest.mark.asyncio
async def test_fallback_final_unrendered_send_does_not_claim_delivery():
    """An ACKed-but-unapplied fallback send must leave flags unset."""
    adapter = _LyingSendAdapter()
    consumer = GatewayStreamConsumer(
        adapter, "chat-1", StreamConsumerConfig(cursor="", edit_interval=0.0)
    )
    consumer._message_id = "m-partial"
    consumer._last_sent_text = "Working on i"
    consumer._visible_prefix_confirmed = True

    await consumer._send_fallback_final("Working on it. Done!")

    assert adapter.sends, "fallback must attempt the tail send"
    assert consumer._final_response_sent is False
    assert consumer._final_content_delivered is False


@pytest.mark.asyncio
async def test_empty_fallback_unrendered_send_returns_failed_not_ambiguous():
    """Known ``content_rendered=False`` must not be treated as ambiguous."""
    adapter = _LyingSendAdapter()
    adapter.RESEND_FINAL_ON_EMPTY_STREAM_FALLBACK = True
    consumer = GatewayStreamConsumer(
        adapter, "chat-1", StreamConsumerConfig(cursor="", edit_interval=0.0)
    )
    # Visible prefix already equals the final text → empty-continuation lane.
    consumer._message_id = "m-preview"
    consumer._last_sent_text = "Complete answer on screen"
    consumer._visible_prefix_confirmed = True

    delivery = await consumer._send_empty_fallback_final("Complete answer on screen")

    assert delivery == "failed"
    assert consumer._final_response_sent is False
    assert consumer._final_content_delivered is False


class _HonestSendAdapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="***"), Platform.TELEGRAM)
        self.sends = []
        self.MAX_MESSAGE_LENGTH = 4096

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def get_chat_info(self, chat_id):
        return {}

    async def send_typing(self, chat_id, metadata=None) -> None:
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        self.sends.append(content)
        return SendResult(success=True, message_id=f"m-{len(self.sends)}")

    async def edit_message(
        self, chat_id, message_id, content, *, finalize: bool = False, metadata=None
    ) -> SendResult:
        return SendResult(success=True, message_id=message_id)

    async def delete_message(self, chat_id, message_id) -> bool:
        return True


@pytest.mark.asyncio
async def test_confirmed_prefix_plus_rendered_tail_claims_delivery():
    """Prefix-preserving tail send with honest transports must suppress."""
    adapter = _HonestSendAdapter()
    consumer = GatewayStreamConsumer(
        adapter, "chat-1", StreamConsumerConfig(cursor="", edit_interval=0.0)
    )
    consumer._message_id = "m-partial"
    consumer._last_sent_text = "Working on i"
    consumer._visible_prefix_confirmed = True

    await consumer._send_fallback_final("Working on it. Done!")

    assert any("Done!" in s for s in adapter.sends)
    assert consumer._final_response_sent is True
    assert consumer._final_content_delivered is True


@pytest.mark.asyncio
async def test_unverified_prefix_equal_to_final_forces_confirmed_full_send():
    """A complete but unverified preview is not enough to suppress the final."""
    adapter = _HonestSendAdapter()
    consumer = GatewayStreamConsumer(
        adapter, "chat-1", StreamConsumerConfig(cursor="", edit_interval=0.0)
    )
    consumer._message_id = "m-preview"
    consumer._last_sent_text = "Complete answer on screen"
    consumer._fallback_prefix = "Complete answer on screen"
    consumer._visible_prefix_confirmed = False  # unverified edit ACK
    consumer._already_sent = True  # state left by the unrendered edit ACK

    await consumer._send_fallback_final("Complete answer on screen")

    assert adapter.sends == ["Complete answer on screen"]
    assert consumer._final_response_sent is True
    assert consumer._final_content_delivered is True
    assert consumer._delivered_final_text == "Complete answer on screen"


@pytest.mark.asyncio
async def test_unverified_empty_fallback_preview_keeps_gateway_send_enabled(monkeypatch):
    """A flood-rejected unverified preview must not become the delivered final."""
    adapter = _HonestSendAdapter()
    adapter.RESEND_FINAL_ON_EMPTY_STREAM_FALLBACK = True
    consumer = GatewayStreamConsumer(
        adapter, "chat-1", StreamConsumerConfig(cursor="", edit_interval=0.0)
    )
    final_text = "Complete answer on screen"
    consumer._message_id = "m-preview"
    consumer._last_sent_text = final_text
    consumer._fallback_prefix = final_text
    consumer._visible_prefix_confirmed = False  # unverified edit ACK
    consumer._already_sent = True  # state left by the unrendered edit ACK

    async def preview_flood(_final_text):
        return "preview"

    monkeypatch.setattr(consumer, "_send_empty_fallback_final", preview_flood)

    await consumer._send_fallback_final(final_text)

    assert adapter.sends == []
    assert consumer._already_sent is False
    assert consumer._final_response_sent is False
    assert consumer._final_content_delivered is False
    assert consumer._delivered_final_text is None


@pytest.mark.asyncio
async def test_unverified_prefix_plus_tail_does_not_claim_delivery():
    """An unverified prefix must force the gateway's corrective full send."""
    adapter = _HonestSendAdapter()
    consumer = GatewayStreamConsumer(
        adapter, "chat-1", StreamConsumerConfig(cursor="", edit_interval=0.0)
    )
    consumer._message_id = "m-partial"
    consumer._last_sent_text = "Working on i"
    consumer._visible_prefix_confirmed = False  # unverified edit ACK

    await consumer._send_fallback_final("Working on it. Done!")

    assert adapter.sends, "tail must still be attempted"
    assert consumer._final_response_sent is False
    assert consumer._final_content_delivered is False


# ---------------------------------------------------------------------------
# Gateway-boundary: reconciliation must honor content_rendered
# ---------------------------------------------------------------------------

STREAMED = "prefix only"
FULL = "prefix only plus the missing tail"


class _CaptureAdapter(BasePlatformAdapter):
    """Records send/edit; reconciliation edits can ACK without rendering."""

    def __init__(self, *, lie_on_content_containing: str | None = None):
        super().__init__(PlatformConfig(enabled=True, token="***"), Platform.TELEGRAM)
        self.sent = []
        self.edits = []
        self._lie_needle = lie_on_content_containing
        self._next_id = 0

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def get_chat_info(self, chat_id):
        return {"id": chat_id}

    async def send_typing(self, chat_id, metadata=None) -> None:
        return None

    async def stop_typing(self, chat_id) -> None:
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        self._next_id += 1
        self.sent.append(content)
        return SendResult(success=True, message_id=f"m-{self._next_id}")

    async def edit_message(
        self, chat_id, message_id, content, *, finalize: bool = False, metadata=None
    ) -> SendResult:
        self.edits.append({"content": content, "finalize": finalize})
        lie = bool(self._lie_needle) and self._lie_needle in content
        return SendResult(
            success=True,
            message_id=message_id,
            content_rendered=not lie,
        )


class _ForcedStaleConsumer(GatewayStreamConsumer):
    """Force the #71643 stale-finalize shape after the stream drain.

    Seeds an editable message id and records only STREAMED as the turn-final
    payload so the gateway's reconciliation edit runs against FULL.
    """

    async def run(self):
        await super().run()
        if not self._message_id or self._message_id == "__no_edit__":
            result = await self.adapter.send(
                chat_id=self.chat_id,
                content=STREAMED,
                metadata=self._metadata_for_send(expect_edits=True),
            )
            self._message_id = result.message_id or "m-forced"
        self._already_sent = True
        self._last_sent_text = STREAMED
        self._visible_prefix_confirmed = True
        self._final_response_sent = True
        self._final_content_delivered = True
        self._record_turn_final_payload(STREAMED)


class _QuietAgent:
    """No stream deltas — the forced consumer supplies the stale preview."""

    def __init__(self, **kwargs):
        self.stream_delta_callback = kwargs.get("stream_delta_callback")
        self.tools = []
        self._final = FULL
        self._transformed = False

    def run_conversation(self, message, conversation_history=None, task_id=None):
        result = {
            "final_response": self._final,
            "response_previewed": False,
            "messages": [],
            "api_calls": 1,
        }
        if self._transformed:
            result["response_transformed"] = True
        return result


class _TransformedQuietAgent(_QuietAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._final = FULL + " [plugin]"
        self._transformed = True


def _make_runner(adapter):
    gateway_run = importlib.import_module("gateway.run")
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.adapters = {adapter.platform: adapter}
    runner._voice_mode = {}
    runner._prefill_messages = []
    runner._ephemeral_system_prompt = ""
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._session_db = None
    runner._running_agents = {}
    runner._session_run_generation = {}
    runner.session_store = SimpleNamespace(_entries={}, _save=lambda: None)
    runner.hooks = SimpleNamespace(loaded_hooks=False)
    runner.config = SimpleNamespace(
        thread_sessions_per_user=False,
        group_sessions_per_user=False,
        stt_enabled=False,
        streaming=StreamingConfig.from_dict(
            {"enabled": True, "edit_interval": 0.01, "buffer_threshold": 1}
        ),
    )
    return runner


async def _run_forced_stale_turn(monkeypatch, tmp_path, agent_cls, adapter, session_id):
    (tmp_path / "config.yaml").write_text(
        yaml.dump(
            {
                "display": {"tool_progress": "off", "interim_assistant_messages": False},
                "streaming": {
                    "enabled": True,
                    "edit_interval": 0.01,
                    "buffer_threshold": 1,
                },
            }
        ),
        encoding="utf-8",
    )
    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)
    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = agent_cls
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

    gateway_run = importlib.import_module("gateway.run")
    stream_consumer_mod = importlib.import_module("gateway.stream_consumer")
    monkeypatch.setattr(
        stream_consumer_mod, "GatewayStreamConsumer", _ForcedStaleConsumer
    )
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"}
    )

    runner = _make_runner(adapter)
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1001",
        chat_type="group",
    )
    return await runner._run_agent(
        message="hi",
        context_prompt="",
        history=[],
        source=source,
        session_id=session_id,
        session_key="agent:main:telegram:group:-1001",
    )


@pytest.mark.asyncio
async def test_stale_reconcile_unrendered_edit_does_not_set_already_sent(
    monkeypatch, tmp_path
):
    """Reconciliation edit ACK without render must leave already_sent unset."""
    adapter = _CaptureAdapter(lie_on_content_containing="missing tail")
    result = await _run_forced_stale_turn(
        monkeypatch,
        tmp_path,
        _QuietAgent,
        adapter,
        "sess-unrendered-stale-reconcile",
    )
    assert result["final_response"] == FULL
    assert any(
        e["content"] == FULL and e["finalize"] for e in adapter.edits
    ), f"expected a reconciliation edit attempt; edits={adapter.edits!r}"
    # Unrendered reconcile must not claim delivery — the normal final send
    # path remains free to deliver.
    assert result.get("already_sent") is not True


@pytest.mark.asyncio
async def test_transformed_reconcile_unrendered_edit_does_not_set_already_sent(
    monkeypatch, tmp_path
):
    """Transformed-response edit without render must not suppress the final."""
    adapter = _CaptureAdapter(lie_on_content_containing="missing tail")
    result = await _run_forced_stale_turn(
        monkeypatch,
        tmp_path,
        _TransformedQuietAgent,
        adapter,
        "sess-unrendered-xform-reconcile",
    )
    assert any("missing tail" in e["content"] for e in adapter.edits), (
        f"expected a transformed reconciliation edit; edits={adapter.edits!r}"
    )
    assert result.get("already_sent") is not True
