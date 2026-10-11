"""``tools.enabled_toolsets`` profile pin must drive runtime toolset resolution."""

import pytest

from hermes_cli.tools_config import (
    _get_platform_tools,
    _profile_toolset_pin,
    _toolsets_from_profile_pin,
)


@pytest.fixture
def pin_config(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    return {
        "platform_toolsets": {"cli": ["hermes-cli"]},
        "tools": {
            "enabled_toolsets": [
                "clarify",
                "file",
                "terminal",
                "memory",
                "web",
            ]
        },
    }


def test_profile_pin_absent_uses_platform_toolsets(monkeypatch):
    cfg = {"platform_toolsets": {"cli": ["memory", "web"]}}
    assert _profile_toolset_pin(cfg) is None
    enabled = _get_platform_tools(cfg, "cli", include_default_mcp_servers=False)
    assert "memory" in enabled
    assert "web" in enabled
    assert "browser" not in enabled


def test_profile_pin_present_restricts_toolsets(pin_config):
    assert _profile_toolset_pin(pin_config) is not None
    enabled = _get_platform_tools(
        pin_config, "cli", include_default_mcp_servers=False
    )
    assert enabled == {"clarify", "file", "terminal", "memory", "web"}
    assert "browser" not in enabled
    assert "vision" not in enabled
    assert "hermes-cli" not in enabled


def test_profile_pin_honors_agent_disabled_toolsets(pin_config):
    pin_config["agent"] = {"disabled_toolsets": ["memory"]}
    enabled = _get_platform_tools(
        pin_config, "cli", include_default_mcp_servers=False
    )
    assert "memory" not in enabled
    assert "file" in enabled


def test_profile_pin_includes_enabled_mcp_by_default(pin_config):
    pin_config["mcp_servers"] = {
        "my-server": {"command": "echo", "args": [], "enabled": True},
    }
    enabled = _get_platform_tools(pin_config, "cli", include_default_mcp_servers=True)
    assert "my-server" in enabled
    assert "file" in enabled


def test_toolsets_from_profile_pin_respects_platform_restrictions():
    pinned = ["discord_admin", "file"]
    enabled = _toolsets_from_profile_pin({}, "cli", pinned, include_default_mcp_servers=False)
    assert "file" in enabled
    assert "discord_admin" not in enabled


def test_resume_rebuild_uses_profile_pin(pin_config):
    """session.resume rebuilds via _make_agent → _load_enabled_toolsets; pin must stick."""
    enabled = _get_platform_tools(pin_config, "cli", include_default_mcp_servers=False)
    # Second resolution models a cold resume agent rebuild on the same profile home.
    enabled_again = _get_platform_tools(pin_config, "cli", include_default_mcp_servers=False)
    assert enabled == enabled_again
    assert "browser" not in enabled


def test_load_enabled_toolsets_skips_coding_posture_when_pinned(monkeypatch, pin_config):
    import agent.coding_context as cc
    import hermes_cli.config as config_mod
    import tui_gateway.server as server

    monkeypatch.delenv("HERMES_TUI_TOOLSETS", raising=False)
    monkeypatch.setattr(cc, "coding_selection", lambda **_: ["coding"])
    monkeypatch.setattr(config_mod, "load_config", lambda: pin_config)

    result = server._load_enabled_toolsets("tui")
    assert result is not None
    assert "coding" not in result
    assert "file" in result
    assert "project" in result
