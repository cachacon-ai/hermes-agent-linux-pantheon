"""Pantheon ``tools.enabled_toolsets`` pin parsing and scoped resolution."""

import logging

import pytest

from hermes_cli.tools_config import (
    _get_platform_tools,
    _normalize_profile_toolset_pin,
    _profile_toolset_pin,
    resolve_pantheon_profile_toolsets,
)


@pytest.fixture
def pin_config(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    return {
        "platform_toolsets": {"cli": ["hermes-cli"], "telegram": ["hermes-telegram"]},
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


def test_profile_pin_absent_uses_platform_toolsets():
    cfg = {"platform_toolsets": {"cli": ["memory", "web"]}}
    assert _profile_toolset_pin(cfg) is None
    enabled = _get_platform_tools(cfg, "cli", include_default_mcp_servers=False)
    assert "memory" in enabled
    assert "web" in enabled
    assert "browser" not in enabled


def test_get_platform_tools_ignores_pin_for_messaging_and_cron(pin_config):
    """Pin must not restrict non-Pantheon-chat platform resolution."""
    pinned_only = resolve_pantheon_profile_toolsets(
        pin_config,
        "cli",
        _profile_toolset_pin(pin_config),
        include_default_mcp_servers=False,
    )
    cli_with_pin = _get_platform_tools(
        pin_config, "cli", include_default_mcp_servers=False
    )
    assert cli_with_pin != pinned_only
    assert len(cli_with_pin) > len(pinned_only)

    telegram = _get_platform_tools(
        pin_config, "telegram", include_default_mcp_servers=False
    )
    assert telegram != pinned_only

    cron = _get_platform_tools(pin_config, "cron", include_default_mcp_servers=False)
    assert cron != pinned_only


def test_normalize_profile_toolset_pin_deduplicates_and_accepts_string(caplog):
    assert _normalize_profile_toolset_pin(["file", "file", " web ", ""]) == [
        "file",
        "web",
    ]
    with caplog.at_level(logging.WARNING):
        assert _normalize_profile_toolset_pin("terminal") == ["terminal"]
    assert "one-item list" in caplog.text

    with caplog.at_level(logging.WARNING):
        assert _normalize_profile_toolset_pin({"bad": True}) is None
    assert "invalid type" in caplog.text


def test_resolve_pantheon_profile_toolsets_restricts_to_pin(pin_config):
    pinned = _profile_toolset_pin(pin_config)
    enabled = resolve_pantheon_profile_toolsets(
        pin_config, "cli", pinned, include_default_mcp_servers=False
    )
    assert enabled == {"clarify", "file", "terminal", "memory", "web"}
    assert "browser" not in enabled


def test_resolve_pantheon_honors_agent_disabled_toolsets(pin_config):
    pin_config["agent"] = {"disabled_toolsets": ["memory"]}
    pinned = _profile_toolset_pin(pin_config)
    enabled = resolve_pantheon_profile_toolsets(
        pin_config, "cli", pinned, include_default_mcp_servers=False
    )
    assert "memory" not in enabled
    assert "file" in enabled


def test_resolve_pantheon_includes_enabled_mcp(pin_config):
    pin_config["mcp_servers"] = {
        "my-server": {"command": "echo", "args": [], "enabled": True},
    }
    pinned = _profile_toolset_pin(pin_config)
    enabled = resolve_pantheon_profile_toolsets(
        pin_config, "cli", pinned, include_default_mcp_servers=True
    )
    assert "my-server" in enabled
    assert "file" in enabled


def test_resolve_pantheon_bad_pin_does_not_fail_open(caplog, pin_config):
    caplog.set_level(logging.WARNING)
    enabled = resolve_pantheon_profile_toolsets(
        pin_config,
        "cli",
        ["not-a-real-toolset", "also-fake"],
        include_default_mcp_servers=False,
    )
    assert enabled == set()
    assert "unknown toolset" in caplog.text
    assert "zero toolsets" in caplog.text


def test_resolve_pantheon_respects_platform_restrictions():
    pinned = ["discord_admin", "file"]
    enabled = resolve_pantheon_profile_toolsets(
        {}, "cli", pinned, include_default_mcp_servers=False
    )
    assert "file" in enabled
    assert "discord_admin" not in enabled
