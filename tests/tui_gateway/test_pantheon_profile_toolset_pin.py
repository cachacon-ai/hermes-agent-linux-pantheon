"""Pantheon profile toolset pin applies only to pantheon chat session start."""

from __future__ import annotations

import types
from unittest.mock import MagicMock, patch

import pytest

import tui_gateway.server as server


@pytest.fixture
def pin_config():
    return {
        "platform_toolsets": {"cli": ["hermes-cli"]},
        "tools": {
            "enabled_toolsets": ["file", "terminal", "memory", "web"],
        },
        "agent": {"system_prompt": ""},
        "model": {"default": "test-model", "provider": "openai"},
    }


@pytest.fixture
def clean_env(monkeypatch):
    monkeypatch.delenv("HERMES_TUI_TOOLSETS", raising=False)
    return monkeypatch


def _make_agent_toolsets(cfg, monkeypatch, *, session_source: str):
    monkeypatch.setattr(server, "_load_cfg", lambda: cfg)
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: cfg,
    )
    fake_runtime = {
        "provider": "openai",
        "base_url": "https://example.com",
        "api_key": "sk-test",
        "api_mode": "chat_completions",
        "command": None,
        "args": None,
        "credential_pool": None,
    }
    with (
        patch("tui_gateway.server._get_db", return_value=MagicMock()),
        patch("tui_gateway.server._load_reasoning_config", return_value=None),
        patch("tui_gateway.server._load_service_tier", return_value=None),
        patch("tui_gateway.server._load_provider_routing", return_value={}),
        patch("tui_gateway.server._resolve_startup_runtime", return_value=("test-model", None)),
        patch(
            "tui_gateway.server._resolve_runtime_with_fallback",
            return_value=MagicMock(runtime=fake_runtime, used_fallback=False, selected_model=None),
        ),
        patch("hermes_cli.mcp_startup.wait_for_mcp_discovery"),
        patch("tui_gateway.entry.wait_for_mcp_discovery"),
        patch("run_agent.AIAgent") as mock_agent,
    ):
        server._make_agent("sid-pin", "key-pin", platform_override=session_source)
    return set(mock_agent.call_args.kwargs.get("enabled_toolsets") or [])


def test_load_enabled_toolsets_scoped_to_pantheon_source(clean_env, pin_config, monkeypatch):
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: pin_config)

    pantheon = set(server._load_enabled_toolsets("pantheon") or [])
    assert "file" in pantheon
    assert "browser" not in pantheon
    assert "project" in pantheon

    tui = set(server._load_enabled_toolsets("tui") or [])
    assert "browser" in tui or "hermes-cli" in str(tui)


def test_make_agent_pantheon_source_honors_pin(clean_env, pin_config, monkeypatch):
    toolsets = _make_agent_toolsets(pin_config, monkeypatch, session_source="pantheon")
    assert "file" in toolsets
    assert "browser" not in toolsets
    assert "project" in toolsets


def test_make_agent_telegram_source_ignores_pin(clean_env, pin_config, monkeypatch):
    toolsets = _make_agent_toolsets(pin_config, monkeypatch, session_source="telegram")
    assert "file" not in toolsets or len(toolsets) > 4


def test_bad_pin_pantheon_minimal_not_hermes_cli(clean_env, pin_config, monkeypatch):
    pin_config["tools"]["enabled_toolsets"] = ["totally-unknown-toolset"]
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: pin_config)
    result = server._load_enabled_toolsets("pantheon")
    assert result is not None
    assert "browser" not in result
    assert "file" not in result
    assert "project" in result


def test_session_create_pantheon_builds_pinned_agent(clean_env, pin_config, monkeypatch):
    """session.create → deferred build must pass pantheon source into toolset resolution."""
    built: dict = {}

    def _capture_build(sid: str, delay: float = 0.05) -> None:
        session = server._sessions.get(sid)
        if session is None:
            return
        toolsets = set(server._load_enabled_toolsets(server._session_source(session)) or [])
        built["toolsets"] = toolsets
        session["agent"] = MagicMock(enabled_toolsets=sorted(toolsets))
        ready = session.get("agent_ready")
        if ready is not None:
            ready.set()

    monkeypatch.setattr(server, "_schedule_agent_build", _capture_build)
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: pin_config)

    out = server._methods["session.create"](
        1,
        {"source": "pantheon", "artifact_delivery_version": 0},
    )
    assert "error" not in out, out
    sid = out["result"]["session_id"]
    server._schedule_agent_build(sid)
    assert "file" in built.get("toolsets", set())
    assert "browser" not in built.get("toolsets", set())
    server._sessions.pop(sid, None)


def test_pantheon_persisted_session_resume_eager_build_honors_pin(tmp_path, monkeypatch):
    """Persist a pantheon row, cold resume via session.resume + eager_build, pin honored."""
    import os
    import uuid

    import yaml
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from hermes_state import SessionDB

    home = tmp_path / ".hermes"
    home.mkdir()
    cfg = {
        "model": {"default": "test-model-live", "provider": "custom:test"},
        "custom_providers": [
            {
                "name": "test",
                "base_url": "https://example.invalid/v1",
                "api_key": "sk-test-not-real",
                "api_mode": "chat_completions",
            }
        ],
        "platform_toolsets": {"cli": ["hermes-cli"]},
        "tools": {"enabled_toolsets": ["file", "terminal", "memory", "web"]},
    }
    (home / "config.yaml").write_text(yaml.safe_dump(cfg))
    monkeypatch.setenv("HERMES_HOME", str(home))
    home_token = set_hermes_home_override(str(home))
    for var in list(os.environ):
        if var.endswith("_API_KEY") or var in ("OPENROUTER_KEY", "NOUS_KEY"):
            monkeypatch.delenv(var, raising=False)

    import hermes_state

    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", home / "state.db")
    monkeypatch.setattr(server, "_db", None, raising=False)
    monkeypatch.setattr(server, "_hermes_home", str(home), raising=False)
    monkeypatch.delenv("HERMES_TUI_TOOLSETS", raising=False)

    sid = uuid.uuid4().hex[:12]
    db = SessionDB(db_path=home / "state.db")
    db.create_session(
        sid,
        source="pantheon",
        model="test-model-live",
        model_config={
            "model": "test-model-live",
            "provider": "custom:test",
            "base_url": "https://example.invalid/v1",
            "api_mode": "chat_completions",
        },
    )
    db.append_message(sid, "user", "hello")
    db.close()

    def _fake_agent(**kwargs):
        return types.SimpleNamespace(
            enabled_toolsets=kwargs.get("enabled_toolsets"),
            model=kwargs.get("model"),
            provider=kwargs.get("provider"),
            session_id=kwargs.get("session_id"),
            reasoning_config=kwargs.get("reasoning_config"),
            service_tier=kwargs.get("service_tier"),
            _memory_manager=MagicMock(),
            _last_flushed_db_idx=0,
            close=lambda: None,
        )

    server._sessions.clear()
    with (
        patch("hermes_cli.mcp_startup.wait_for_mcp_discovery"),
        patch("tui_gateway.entry.wait_for_mcp_discovery"),
        patch("run_agent.AIAgent", side_effect=_fake_agent),
    ):
        resp = server.handle_request(
            {
                "id": "resume-pantheon-pin",
                "method": "session.resume",
                "params": {
                    "session_id": sid,
                    "source": "pantheon",
                    "eager_build": True,
                    "omit_messages": True,
                },
            }
        )
    try:
        assert "error" not in resp, resp.get("error")
        live_sid = resp["result"]["session_id"]
        live = server._sessions[live_sid]
        agent = live["agent"]
        assert agent is not None
        enabled = set(getattr(agent, "enabled_toolsets", []) or [])
        assert "file" in enabled
        assert "browser" not in enabled
        assert "project" in enabled
    finally:
        live_sid = (resp.get("result") or {}).get("session_id")
        if live_sid:
            server.handle_request(
                {
                    "id": "close-pantheon",
                    "method": "session.close",
                    "params": {"session_id": live_sid},
                }
            )
        reset_hermes_home_override(home_token)


def test_tui_db_session_resume_pantheon_honors_pin(tmp_path, monkeypatch):
    """Desktop/TUI-born row; Pantheon resume must apply profile pin for this turn."""
    import os
    import uuid

    import yaml
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from hermes_state import SessionDB

    home = tmp_path / ".hermes"
    home.mkdir()
    cfg = {
        "model": {"default": "test-model-live", "provider": "custom:test"},
        "custom_providers": [
            {
                "name": "test",
                "base_url": "https://example.invalid/v1",
                "api_key": "sk-test-not-real",
                "api_mode": "chat_completions",
            }
        ],
        "platform_toolsets": {"cli": ["hermes-cli"]},
        "tools": {"enabled_toolsets": ["file", "terminal", "memory", "web"]},
    }
    (home / "config.yaml").write_text(yaml.safe_dump(cfg))
    monkeypatch.setenv("HERMES_HOME", str(home))
    home_token = set_hermes_home_override(str(home))
    for var in list(os.environ):
        if var.endswith("_API_KEY") or var in ("OPENROUTER_KEY", "NOUS_KEY"):
            monkeypatch.delenv(var, raising=False)

    import hermes_state

    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", home / "state.db")
    monkeypatch.setattr(server, "_db", None, raising=False)
    monkeypatch.setattr(server, "_hermes_home", str(home), raising=False)
    monkeypatch.delenv("HERMES_TUI_TOOLSETS", raising=False)

    sid = uuid.uuid4().hex[:12]
    db = SessionDB(db_path=home / "state.db")
    db.create_session(
        sid,
        source="desktop",
        model="test-model-live",
        model_config={
            "model": "test-model-live",
            "provider": "custom:test",
            "base_url": "https://example.invalid/v1",
            "api_mode": "chat_completions",
        },
    )
    db.close()

    def _fake_agent(**kwargs):
        return types.SimpleNamespace(
            enabled_toolsets=kwargs.get("enabled_toolsets"),
            model=kwargs.get("model"),
            provider=kwargs.get("provider"),
            session_id=kwargs.get("session_id"),
            _memory_manager=MagicMock(),
            _last_flushed_db_idx=0,
            close=lambda: None,
        )

    server._sessions.clear()
    with (
        patch("hermes_cli.mcp_startup.wait_for_mcp_discovery"),
        patch("tui_gateway.entry.wait_for_mcp_discovery"),
        patch("run_agent.AIAgent", side_effect=_fake_agent),
    ):
        resp = server.handle_request(
            {
                "id": "resume-tui-as-pantheon",
                "method": "session.resume",
                "params": {
                    "session_id": sid,
                    "source": "pantheon",
                    "eager_build": True,
                    "omit_messages": True,
                },
            }
        )
    try:
        assert "error" not in resp, resp.get("error")
        live_sid = resp["result"]["session_id"]
        live = server._sessions[live_sid]
        assert live.get("source") == "desktop"
        assert live.get("client_source") == "pantheon"
        enabled = set(getattr(live["agent"], "enabled_toolsets", []) or [])
        assert "file" in enabled
        assert "browser" not in enabled
    finally:
        live_sid = (resp.get("result") or {}).get("session_id")
        if live_sid:
            server.handle_request(
                {
                    "id": "close-cross",
                    "method": "session.close",
                    "params": {"session_id": live_sid},
                }
            )
        reset_hermes_home_override(home_token)


def test_live_desktop_session_resume_pantheon_rebuilds_pin(tmp_path, monkeypatch, pin_config):
    """Fast-path reuse: Pantheon resume rebinds client surface and toolsets."""
    import os

    import yaml
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from hermes_state import SessionDB

    home = tmp_path / ".hermes"
    home.mkdir()
    cfg = dict(pin_config)
    cfg.update(
        {
            "model": {"default": "test-model-live", "provider": "custom:test"},
            "custom_providers": [
                {
                    "name": "test",
                    "base_url": "https://example.invalid/v1",
                    "api_key": "sk-test-not-real",
                    "api_mode": "chat_completions",
                }
            ],
        }
    )
    (home / "config.yaml").write_text(yaml.safe_dump(cfg))
    monkeypatch.setenv("HERMES_HOME", str(home))
    home_token = set_hermes_home_override(str(home))
    for var in list(os.environ):
        if var.endswith("_API_KEY") or var in ("OPENROUTER_KEY", "NOUS_KEY"):
            monkeypatch.delenv(var, raising=False)

    import hermes_state

    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", home / "state.db")
    monkeypatch.setattr(server, "_db", None, raising=False)
    monkeypatch.setattr(server, "_hermes_home", str(home), raising=False)
    monkeypatch.delenv("HERMES_TUI_TOOLSETS", raising=False)

    stored = "stored-desktop-chat"
    db = SessionDB(db_path=home / "state.db")
    db.create_session(
        stored,
        source="desktop",
        model="test-model-live",
        model_config={
            "model": "test-model-live",
            "provider": "custom:test",
            "base_url": "https://example.invalid/v1",
            "api_mode": "chat_completions",
        },
    )
    db.close()

    agents_built: list[set[str]] = []

    def _fake_agent(**kwargs):
        toolsets = set(kwargs.get("enabled_toolsets") or [])
        agents_built.append(toolsets)
        return types.SimpleNamespace(
            enabled_toolsets=list(toolsets),
            model=kwargs.get("model"),
            provider=kwargs.get("provider"),
            session_id=kwargs.get("session_id"),
            _memory_manager=MagicMock(),
            _last_flushed_db_idx=0,
            close=lambda: None,
        )

    server._sessions.clear()
    with (
        patch("hermes_cli.mcp_startup.wait_for_mcp_discovery"),
        patch("tui_gateway.entry.wait_for_mcp_discovery"),
        patch("run_agent.AIAgent", side_effect=_fake_agent),
    ):
        first = server.handle_request(
            {
                "id": "resume-desktop",
                "method": "session.resume",
                "params": {
                    "session_id": stored,
                    "source": "desktop",
                    "eager_build": True,
                    "omit_messages": True,
                },
            }
        )
        assert "error" not in first, first.get("error")
        live_sid = first["result"]["session_id"]
        assert agents_built
        assert "browser" in agents_built[0] or len(agents_built[0]) > 5

        second = server.handle_request(
            {
                "id": "resume-pantheon-fast",
                "method": "session.resume",
                "params": {
                    "session_id": stored,
                    "source": "pantheon",
                    "eager_build": True,
                    "omit_messages": True,
                },
            }
        )
    try:
        assert "error" not in second, second.get("error")
        assert second["result"]["session_id"] == live_sid
        live = server._sessions[live_sid]
        assert live.get("client_source") == "pantheon"
        enabled = set(getattr(live["agent"], "enabled_toolsets", []) or [])
        assert "file" in enabled
        assert "browser" not in enabled
        assert len(agents_built) >= 2
    finally:
        server.handle_request(
            {
                "id": "close-fast",
                "method": "session.close",
                "params": {"session_id": live_sid},
            }
        )
        reset_hermes_home_override(home_token)


def test_tui_persisted_session_resume_ignores_profile_pin(tmp_path, monkeypatch):
    """Same profile pin in config; non-pantheon resume keeps platform toolsets."""
    import os
    import uuid

    import yaml
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from hermes_state import SessionDB

    home = tmp_path / ".hermes"
    home.mkdir()
    cfg = {
        "model": {"default": "test-model-live", "provider": "custom:test"},
        "custom_providers": [
            {
                "name": "test",
                "base_url": "https://example.invalid/v1",
                "api_key": "sk-test-not-real",
                "api_mode": "chat_completions",
            }
        ],
        "platform_toolsets": {"cli": ["hermes-cli"]},
        "tools": {"enabled_toolsets": ["file", "terminal", "memory", "web"]},
    }
    (home / "config.yaml").write_text(yaml.safe_dump(cfg))
    monkeypatch.setenv("HERMES_HOME", str(home))
    home_token = set_hermes_home_override(str(home))
    for var in list(os.environ):
        if var.endswith("_API_KEY") or var in ("OPENROUTER_KEY", "NOUS_KEY"):
            monkeypatch.delenv(var, raising=False)

    import hermes_state

    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", home / "state.db")
    monkeypatch.setattr(server, "_db", None, raising=False)
    monkeypatch.setattr(server, "_hermes_home", str(home), raising=False)
    monkeypatch.delenv("HERMES_TUI_TOOLSETS", raising=False)

    sid = uuid.uuid4().hex[:12]
    db = SessionDB(db_path=home / "state.db")
    db.create_session(
        sid,
        source="tui",
        model="test-model-live",
        model_config={
            "model": "test-model-live",
            "provider": "custom:test",
            "base_url": "https://example.invalid/v1",
            "api_mode": "chat_completions",
        },
    )
    db.close()

    def _fake_agent(**kwargs):
        return types.SimpleNamespace(
            enabled_toolsets=kwargs.get("enabled_toolsets"),
            model=kwargs.get("model"),
            provider=kwargs.get("provider"),
            session_id=kwargs.get("session_id"),
            _memory_manager=MagicMock(),
            _last_flushed_db_idx=0,
            close=lambda: None,
        )

    server._sessions.clear()
    with (
        patch("hermes_cli.mcp_startup.wait_for_mcp_discovery"),
        patch("tui_gateway.entry.wait_for_mcp_discovery"),
        patch("run_agent.AIAgent", side_effect=_fake_agent),
    ):
        resp = server.handle_request(
            {
                "id": "resume-tui-pin",
                "method": "session.resume",
                "params": {
                    "session_id": sid,
                    "source": "tui",
                    "eager_build": True,
                    "omit_messages": True,
                },
            }
        )
    try:
        assert "error" not in resp, resp.get("error")
        live_sid = resp["result"]["session_id"]
        agent = server._sessions[live_sid]["agent"]
        enabled = set(getattr(agent, "enabled_toolsets", []) or [])
        pinned_only = {"file", "terminal", "memory", "web", "project"}
        assert enabled != pinned_only
        assert "browser" in enabled or len(enabled) > len(pinned_only)
    finally:
        live_sid = (resp.get("result") or {}).get("session_id")
        if live_sid:
            server.handle_request(
                {
                    "id": "close-tui",
                    "method": "session.close",
                    "params": {"session_id": live_sid},
                }
            )
        reset_hermes_home_override(home_token)
