"""PAN-17: per-turn clock context via ``prompt.submit`` (Pantheon / tui_gateway).

Pantheon sends device-local time on every LLM turn as ``clock_context``. Hermes
delivers it to the model through the same ``api_content`` sidecar channel as
gateway must-deliver notes, without putting the stamp in durable ``content``.

RPC contract (Pantheon → ``hermes serve`` / tui_gateway)::

    {
      "method": "prompt.submit",
      "params": {
        "session_id": "<runtime session id from session.create>",
        "text": "<user message>",
        "clock_context": "<optional per-turn clock line, e.g. IANA tz + local time>"
      }
    }

``clock_context`` is optional, a single string (whitespace trimmed; empty
omitted). It is **not** persisted in the user-visible transcript ``content``;
clients loading ``session.history`` / ``_history_to_messages`` see only
``text`` from ``content``. The model receives the stamp on the current turn's
user message via ``api_content`` (or a trailing text part on multimodal turns).
"""

from __future__ import annotations

import threading
import types
from unittest.mock import patch

import pytest

from tui_gateway import server


class _InlineThread:
    def __init__(self, target=None, daemon=None, args=(), kwargs=None):
        self._target = target
        self._args = args
        self._kwargs = kwargs or {}

    def start(self):
        if self._target is not None:
            self._target(*self._args, **self._kwargs)

    def is_alive(self):
        return False

    def join(self, timeout=None):
        return None


CLOCK_LINE = (
    "[Current local time: Wednesday, 2026-10-07 15:00:00 "
    "America/Los_Angeles (UTC-07:00)]"
)
USER_TEXT = "What day is it?"


@pytest.fixture()
def turn_env(monkeypatch, tmp_path):
    monkeypatch.setattr(server.threading, "Thread", _InlineThread)
    monkeypatch.setattr(server, "_wire_callbacks", lambda sid: None)
    monkeypatch.setattr(server, "_sync_agent_model_with_config", lambda sid, session: None)
    monkeypatch.setattr(server, "_session_cwd", lambda session: str(tmp_path))
    monkeypatch.setattr(server, "_register_session_cwd", lambda session: None)
    monkeypatch.setattr(server, "_tts_stream_begin", lambda: None)
    monkeypatch.setattr(server, "_sync_session_key_after_compress", lambda *a, **k: None)
    monkeypatch.setattr(server, "_get_usage", lambda agent: {})
    monkeypatch.setattr(server, "_start_usage_ticker", lambda *a, **k: (threading.Event(), _InlineThread()))
    monkeypatch.setattr(server, "_load_interim_assistant_messages", lambda: False)
    monkeypatch.setattr(server, "_emit", lambda *a, **k: None)
    monkeypatch.setattr(server, "_ensure_active_session_slot", lambda sid, session: None)


def _session(agent, **extra):
    return {
        "agent": agent,
        "session_key": "session-key",
        "history": [],
        "history_lock": threading.Lock(),
        "history_version": 0,
        "running": True,
        "attached_images": [],
        "image_counter": 0,
        "cols": 80,
        **extra,
    }


def test_run_prompt_submit_stages_clock_context_for_sidecar(turn_env):
    """``clock_context`` is staged on the agent for turn_context consumption."""
    seen = {}

    def run_conversation(user_message, **kwargs):
        seen["notes"] = getattr(agent, "_gateway_turn_context_notes", "")
        seen["user_message"] = user_message
        return {
            "final_response": "Tuesday",
            "messages": [
                {"role": "user", "content": USER_TEXT},
                {"role": "assistant", "content": "Tuesday"},
            ],
        }

    agent = types.SimpleNamespace(
        session_id="session-key",
        run_conversation=run_conversation,
        clear_interrupt=lambda: None,
    )
    session = _session(agent)

    server._run_prompt_submit(
        "rid",
        "sid",
        session,
        USER_TEXT,
        clock_context=CLOCK_LINE,
    )

    assert seen["notes"] == CLOCK_LINE
    assert CLOCK_LINE not in seen["user_message"]
    assert CLOCK_LINE not in session["history"][0]["content"]


def test_clock_context_reaches_api_content_not_stored_content():
    """Turn prologue: staged clock rides ``api_content``, not ``content``."""
    from tests.agent.test_gateway_turn_sidecar import _FakeAgent, _build

    agent = _FakeAgent()
    agent._gateway_turn_context_notes = CLOCK_LINE
    with patch("hermes_cli.plugins.invoke_hook", return_value=[]):
        ctx = _build(agent, user_message=USER_TEXT)
    msg = ctx.messages[ctx.current_turn_user_idx]
    assert msg["content"] == USER_TEXT
    assert CLOCK_LINE in msg["api_content"]
    assert CLOCK_LINE not in msg["content"]
