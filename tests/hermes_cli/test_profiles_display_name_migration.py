"""migrate_ui_meta_display_names_to_profile_yaml (PAN-14 slice 2)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from hermes_cli.profiles import (
    create_profile,
    migrate_ui_meta_display_names_to_profile_yaml,
    read_profile_meta,
)


@pytest.fixture
def profile_env(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    default_home = tmp_path / ".hermes"
    default_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(default_home))
    return tmp_path


def test_migration_copies_ui_meta_display_name_once(profile_env):
    home = profile_env / ".hermes"
    meta_path = home / "profile.yaml"
    meta_path.write_text(
        yaml.safe_dump(
            {
                "ui_meta": {"display_name": "Legacy Name", "pet": "cat"},
            }
        ),
        encoding="utf-8",
    )

    first = migrate_ui_meta_display_names_to_profile_yaml()
    assert first["migrated"] == ["default"]
    assert first["errors"] == []
    meta = read_profile_meta(home)
    assert meta["display_name"] == "Legacy Name"

    with open(meta_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    assert raw["ui_meta"]["display_name"] == "Legacy Name"

    second = migrate_ui_meta_display_names_to_profile_yaml()
    assert second["migrated"] == []
    assert "default" in second["skipped"]


def test_migration_named_profile_and_dry_run(profile_env):
    path = create_profile("worker")
    meta_path = path / "profile.yaml"
    meta_path.write_text(
        yaml.safe_dump({"ui_meta": {"display_name": "Worker One"}}),
        encoding="utf-8",
    )

    dry = migrate_ui_meta_display_names_to_profile_yaml(dry_run=True)
    assert dry["migrated"] == ["worker"]
    assert read_profile_meta(path)["display_name"] == ""

    live = migrate_ui_meta_display_names_to_profile_yaml()
    assert live["migrated"] == ["worker"]
    assert read_profile_meta(path)["display_name"] == "Worker One"
