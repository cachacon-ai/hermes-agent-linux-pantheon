"""profiles.configure / profiles.describe display_name (PAN-14 slice 2)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

import tui_gateway.server as srv
from hermes_cli.profiles import create_profile, get_profile_dir


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    return hermes_home


def _configure(name: str, **params):
    return srv._methods["profiles.configure"](
        "configure", {"name": name, **params}
    )["result"]


def _describe(name: str):
    return srv._methods["profiles.describe"]("describe", {"name": name})["result"]


def _list_row(name: str):
    rows = srv._methods["profiles.list"](
        "list", {"include_sessions": False}
    )["result"]["profiles"]
    return next(row for row in rows if row["name"] == name)


def test_display_name_round_trips_configure_describe_list(home):
    create_profile("bot-a", description="worker")
    result = _configure("bot-a", display_name="  Friendly Bot  ")

    assert result["applied"]["display_name"] is True
    assert result["display_name"] == "Friendly Bot"
    assert get_profile_dir("bot-a").name == "bot-a"

    described = _describe("bot-a")
    assert described["display_name"] == "Friendly Bot"
    assert described["ui_meta"] == {}
    assert described["has_avatar"] is False

    listed = _list_row("bot-a")
    assert listed["display_name"] == "Friendly Bot"


def test_display_name_clear_with_empty_string(home):
    _configure("default", display_name="Temporary")
    cleared = _configure("default", display_name="")

    assert cleared["applied"]["display_name"] is True
    assert cleared["display_name"] == ""
    assert _describe("default")["display_name"] == ""


def test_display_name_rejects_too_long(home):
    too_long = "x" * 65
    result = _configure("default", display_name=too_long)

    assert result["applied"]["display_name"] is False
    assert "display_name" not in result
    assert _describe("default")["display_name"] == ""


def test_display_name_rejects_non_string(home):
    result = _configure("default", display_name=123)

    assert result["applied"]["display_name"] is False


def test_soul_write_failure_preserves_original(home, monkeypatch):
    soul_path = home / "SOUL.md"
    soul_path.write_text("keep this text", encoding="utf-8")

    def _fail_atomic_write(path, content, **kwargs):
        raise OSError("simulated mid-write failure")

    import utils

    monkeypatch.setattr(utils, "atomic_write_text", _fail_atomic_write)

    result = _configure("default", soul="replacement would truncate")

    assert result["applied"]["soul"] is False
    assert soul_path.read_text(encoding="utf-8") == "keep this text"


def test_gateway_capabilities_advertises_profile_display_name(home):
    caps = srv._methods["gateway.capabilities"]("caps", {})["result"]
    assert caps.get("profile_display_name") is True
    assert "per_session_exclusive_submit" in caps


def test_describe_ui_meta_and_has_avatar(home):
    meta_path = home / "profile.yaml"
    meta_path.write_text(
        yaml.safe_dump({"ui_meta": {"accent": "#abc"}, "display_name": "Main"}),
        encoding="utf-8",
    )
    assets = home / "assets"
    assets.mkdir()
    (assets / "avatar.png").write_bytes(b"\x89PNG")

    described = _describe("default")
    assert described["display_name"] == "Main"
    assert described["ui_meta"] == {"accent": "#abc"}
    assert described["has_avatar"] is True
