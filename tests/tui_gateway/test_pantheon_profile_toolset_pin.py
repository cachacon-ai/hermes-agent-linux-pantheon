"""Pantheon profile toolset pin applies only to pantheon chat session start."""

from __future__ import annotations

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


def test_session_resume_pantheon_rebuild_honors_pin(clean_env, pin_config, monkeypatch):
    """Cold resume rebuild uses the same pantheon-scoped resolver as create."""
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: pin_config)

    session = {
        "session_key": "resume-key",
        "source": "pantheon",
        "agent": None,
        "agent_ready": __import__("threading").Event(),
        "history": [],
        "profile_home": None,
    }
    toolsets = set(server._load_enabled_toolsets(server._session_source(session)) or [])
    assert "file" in toolsets
    assert "browser" not in toolsets
