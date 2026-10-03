"""session.history must load on-disk transcripts after a gateway restart.

Pantheon (and other clients) keep the durable state.db session id and call
``session.history`` with that id. Before the cold-load path, a restarted
gateway returned 4001/4007 "session not found" even when hundreds of messages
remained in state.db.
"""

from __future__ import annotations

import importlib
import uuid
from unittest.mock import MagicMock, patch

import pytest

from hermes_state import SessionDB


@pytest.fixture()
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(__import__("pathlib").Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def _import_tui_server():
    with patch.dict(
        __import__("sys").modules,
        {"hermes_cli.env_loader": MagicMock(), "hermes_cli.banner": MagicMock()},
    ):
        import tui_gateway.server as server

        return importlib.reload(server)


@pytest.fixture()
def gateway(hermes_home, monkeypatch):
    server = _import_tui_server()
    import hermes_state

    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", hermes_home / "state.db")
    monkeypatch.setattr(server, "_db", None, raising=False)
    monkeypatch.setattr(server, "_db_error", None, raising=False)
    monkeypatch.setattr(server, "_hermes_home", str(hermes_home), raising=False)
    with server._sessions_lock:
        server._sessions.clear()
    yield server
    with server._sessions_lock:
        server._sessions.clear()


def _seed_session(db: SessionDB, *, title: str = "Bot Chat") -> str:
    sid = f"20260903_{uuid.uuid4().hex[:6]}_897200"
    db.create_session(sid, source="desktop", model="test-model")
    db.set_session_title(sid, title)
    db.append_message(sid, "user", "hello from disk")
    db.append_message(sid, "assistant", "still here")
    return sid


def test_session_history_loads_persisted_session_without_runtime(gateway, hermes_home):
    db = SessionDB(db_path=hermes_home / "state.db")
    stored_id = _seed_session(db)
    db.close()

    assert gateway._sessions == {}

    resp = gateway.handle_request(
        {
            "id": "hist",
            "method": "session.history",
            "params": {"session_id": stored_id},
        }
    )
    assert "error" not in resp, resp
    assert resp["result"]["count"] == 2
    roles = [m["role"] for m in resp["result"]["messages"]]
    assert roles == ["user", "assistant"]


def test_session_history_still_404s_when_row_absent(gateway):
    resp = gateway.handle_request(
        {
            "id": "hist",
            "method": "session.history",
            "params": {"session_id": "20260903_000000_missing"},
        }
    )
    assert resp.get("error", {}).get("code") == 4007


def test_session_history_resolves_live_session_by_stored_key(gateway, hermes_home):
    db = SessionDB(db_path=hermes_home / "state.db")
    stored_id = _seed_session(db)
    db.close()

    runtime_id = "rt-live-1"
    gateway._sessions[runtime_id] = {
        "session_key": stored_id,
        "history": [{"role": "user", "content": "stale memory"}],
        "history_lock": __import__("threading").Lock(),
        "running": False,
        "agent": None,
        "created_at": 1.0,
        "last_active": 1.0,
    }

    class _Db:
        def get_messages_as_conversation(self, _key, include_ancestors=True, include_row_ids=False, **_kw):
            return [
                {"role": "user", "content": "hello from disk", "_row_id": 1},
                {"role": "assistant", "content": "still here", "_row_id": 2},
            ]

    gateway._get_db = lambda: _Db()

    resp = gateway.handle_request(
        {
            "id": "hist",
            "method": "session.history",
            "params": {"session_id": stored_id},
        }
    )
    assert "error" not in resp, resp
    assert resp["result"]["count"] == 2
