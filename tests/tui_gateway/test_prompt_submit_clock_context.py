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
        "clock_context": {
          "iana_timezone": "America/New_York",
          "formatted": "Current local time: Wed, Oct 7, 2026, 2:47 PM EDT (America/New_York)"
        }
      }
    }

``clock_context`` is optional. Accept either a non-empty string (whitespace
trimmed) or an object with a non-empty ``formatted`` string (Pantheon also
sends ``iana_timezone``; Hermes uses ``formatted`` for the model). Empty values
are omitted. It is **not** persisted in the user-visible transcript ``content``;
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
from tui_gateway.methods_prompt import normalize_prompt_clock_context


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
PANTHEON_CLOCK = {
    "iana_timezone": "America/New_York",
    "formatted": "Current local time: Wed, Oct 7, 2026, 2:47 PM EDT (America/New_York)",
}
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


def test_normalize_prompt_clock_context_accepts_pantheon_object():
    assert normalize_prompt_clock_context(PANTHEON_CLOCK) == PANTHEON_CLOCK["formatted"]
    assert normalize_prompt_clock_context(CLOCK_LINE) == CLOCK_LINE
    assert normalize_prompt_clock_context({"formatted": "  "}) is None
    assert normalize_prompt_clock_context({"iana_timezone": "UTC"}) is None


def test_run_prompt_submit_stages_pantheon_object_clock_context(turn_env):
    """Pantheon object shape → ``formatted`` staged for turn_context."""
    seen = {}

    def run_conversation(user_message, **kwargs):
        seen["staged"] = getattr(agent, "_prompt_clock_context", "")
        return {
            "final_response": "Wednesday",
            "messages": [
                {"role": "user", "content": USER_TEXT},
                {"role": "assistant", "content": "Wednesday"},
            ],
        }

    agent = types.SimpleNamespace(
        session_id="session-key",
        run_conversation=run_conversation,
        clear_interrupt=lambda: None,
    )
    session = _session(agent)
    clock_context = normalize_prompt_clock_context(PANTHEON_CLOCK)

    server._run_prompt_submit(
        "rid",
        "sid",
        session,
        USER_TEXT,
        clock_context=clock_context,
    )

    assert seen["staged"] == PANTHEON_CLOCK["formatted"]
    assert PANTHEON_CLOCK["formatted"] not in session["history"][0]["content"]


def test_run_prompt_submit_stages_clock_context_for_sidecar(turn_env):
    """``clock_context`` is staged on the agent for turn_context consumption."""
    seen = {}

    def run_conversation(user_message, **kwargs):
        seen["staged"] = getattr(agent, "_prompt_clock_context", "")
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

    assert seen["staged"] == CLOCK_LINE
    assert CLOCK_LINE not in seen["user_message"]
    assert CLOCK_LINE not in session["history"][0]["content"]


def test_clock_context_reaches_api_content_not_stored_content():
    """Turn prologue: staged clock rides ``api_content``, not ``content``."""
    from tests.agent.test_gateway_turn_sidecar import _FakeAgent, _build

    agent = _FakeAgent()
    agent._prompt_clock_context = CLOCK_LINE
    with patch("hermes_cli.plugins.invoke_hook", return_value=[]):
        ctx = _build(agent, user_message=USER_TEXT)
    msg = ctx.messages[ctx.current_turn_user_idx]
    assert msg["content"] == USER_TEXT
    assert CLOCK_LINE in msg["api_content"]
    assert CLOCK_LINE not in msg["content"]


def test_multimodal_clock_stays_off_durable_content():
    """Multimodal turns: clock is API-wire only, not appended to stored content."""
    from agent.conversation_loop import append_prompt_clock_to_multimodal_api_content
    from agent.turn_context import append_notes_to_multimodal_content
    from tests.agent.test_gateway_turn_sidecar import _FakeAgent, _build

    multimodal = [
        {"type": "text", "text": "describe this"},
        {"type": "image_url", "image_url": {"url": "https://example.com/x.png"}},
    ]
    agent = _FakeAgent()
    agent._prompt_clock_context = CLOCK_LINE
    with patch("hermes_cli.plugins.invoke_hook", return_value=[]):
        ctx = _build(agent, user_message=multimodal)
    msg = ctx.messages[ctx.current_turn_user_idx]
    assert msg["content"] == multimodal
    assert "api_content" not in msg
    wire = append_prompt_clock_to_multimodal_api_content(multimodal, CLOCK_LINE)
    assert wire[-1] == {"type": "text", "text": CLOCK_LINE}
    # Gateway must-deliver notes still use the durable multimodal append path.
    durable = list(multimodal)
    append_notes_to_multimodal_content(durable, CLOCK_LINE)
    assert durable[-1] == {"type": "text", "text": CLOCK_LINE}
