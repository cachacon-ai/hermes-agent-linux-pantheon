"""Pantheon new-bot plan: Hermes-side resume/profile/session fixes."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from hermes_cli.config import clear_model_endpoint_credentials, get_config_path, save_config
from hermes_state import SessionDB


@pytest.fixture()
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def _import_tui_server():
    with patch.dict(
        sys.modules,
        {"hermes_cli.env_loader": MagicMock(), "hermes_cli.banner": MagicMock()},
    ):
        import tui_gateway.server as server

        return importlib.reload(server)


class TestProfileCreateModelPinFailClosed:
    def test_profiles_create_surfaces_model_write_failure(self, tmp_path, monkeypatch):
        profile_dir = tmp_path / ".hermes" / "profiles" / "bot-a"
        profile_dir.mkdir(parents=True)

        server = _import_tui_server()

        monkeypatch.setattr("hermes_cli.profiles.create_profile", lambda **k: profile_dir)
        monkeypatch.setattr("hermes_cli.profiles.seed_profile_skills", lambda *a, **k: None)
        monkeypatch.setattr("hermes_cli.profiles.check_alias_collision", lambda _n: None)
        monkeypatch.setattr("hermes_cli.profiles.create_wrapper_script", lambda _n: None)

        with patch(
            "hermes_cli.web_routers.profiles._write_profile_model",
            side_effect=RuntimeError("disk full"),
        ):
            out = server.handle_request(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "profiles.create",
                    "params": {
                        "name": "bot-a",
                        "model": "qwen3.8-flash-next",
                        "provider": "engramhalo",
                        "mirror_credentials": False,
                    },
                }
            )

        assert out.get("error") is not None
        assert "model assignment failed" in str(out["error"]["message"]).lower()

    def test_rest_profiles_create_raises_on_model_write_failure(self, tmp_path):
        import asyncio

        from fastapi import HTTPException

        from hermes_cli.web_models import ProfileCreate
        from hermes_cli.web_routers import profiles as profiles_router

        profile_dir = tmp_path / ".hermes" / "profiles" / "bot-b"
        profile_dir.mkdir(parents=True)
        body = ProfileCreate(
            name="bot-b",
            provider="engramhalo",
            model="qwen3.8-flash-next",
        )

        async def run():
            return await profiles_router.create_profile_endpoint(body)

        with patch("hermes_cli.profiles.create_profile", return_value=profile_dir):
            with patch("hermes_cli.profiles.seed_profile_skills"):
                with patch("hermes_cli.profiles.check_alias_collision", return_value=None):
                    with patch("hermes_cli.profiles.create_wrapper_script"):
                        with patch(
                            "hermes_cli.web_routers.profiles._write_profile_model",
                            side_effect=OSError("write failed"),
                        ):
                            with pytest.raises(HTTPException) as exc:
                                asyncio.run(run())
        assert exc.value.status_code == 500
        assert "model assignment failed" in str(exc.value.detail).lower()


class TestProviderSwitchClearsKeyEnv:
    def test_clear_model_endpoint_credentials_pops_key_env(self):
        model_cfg = {
            "provider": "openrouter",
            "default": "anthropic/claude-sonnet-4.6",
            "key_env": "OPENROUTER_API_KEY",
        }
        clear_model_endpoint_credentials(model_cfg, clear_api_key=False, clear_key_env=True)
        assert "key_env" not in model_cfg

    def test_provider_switch_invokes_clear_key_env(self):
        model_cfg = {
            "provider": "engramhalo",
            "default": "qwen3.8-flash-next",
            "key_env": "ENGRAMHALO_API_KEY",
        }
        prev_provider = str(model_cfg.get("provider") or "").strip().lower()
        new_provider = "ollama-cloud"
        if new_provider != prev_provider:
            clear_model_endpoint_credentials(
                model_cfg, clear_api_key=False, clear_key_env=True
            )
        assert "key_env" not in model_cfg


class TestColdResumeReasoningRestore:
    def test_normal_row_restores_reasoning_override(self):
        server = _import_tui_server()
        _stored_session_runtime_overrides = server._stored_session_runtime_overrides

        row = {
            "model": "glm-5.1",
            "model_config": json.dumps(
                {
                    "model": "glm-5.1",
                    "provider": "ollama-cloud",
                    "reasoning_config": {"enabled": True, "effort": "medium"},
                }
            ),
        }
        overrides = _stored_session_runtime_overrides(row)
        assert overrides["reasoning_config_override"] == {
            "enabled": True,
            "effort": "medium",
        }

    def test_bot_chat_restores_reasoning_without_model_pin(self):
        server = _import_tui_server()
        _stored_session_runtime_overrides = server._stored_session_runtime_overrides

        row = {
            "title": "Bot Chat",
            "model": "qwen3.8-flash-next",
            "billing_provider": "engramhalo",
            "model_config": json.dumps(
                {
                    "model": "qwen3.8-flash-next",
                    "provider": "engramhalo",
                    "follow_profile_config": True,
                    "reasoning_config": {"enabled": True, "effort": "xhigh"},
                }
            ),
        }
        overrides = _stored_session_runtime_overrides(row)
        assert "model_override" not in overrides
        assert overrides["reasoning_config_override"] == {
            "enabled": True,
            "effort": "xhigh",
        }

    def test_sync_create_reasoning_from_resume_overrides(self):
        server = _import_tui_server()
        _sync_create_reasoning_override_from_resume_overrides = (
            server._sync_create_reasoning_override_from_resume_overrides
        )

        session = {
            "resume_runtime_overrides": {
                "reasoning_config_override": {"enabled": True, "effort": "low"},
            }
        }
        _sync_create_reasoning_override_from_resume_overrides(session)
        assert session["create_reasoning_override"] == {"enabled": True, "effort": "low"}

    def test_sync_does_not_touch_profile_config(self, hermes_home):
        server = _import_tui_server()
        _sync_create_reasoning_override_from_resume_overrides = (
            server._sync_create_reasoning_override_from_resume_overrides
        )
        save_config({"agent": {"reasoning_effort": "medium"}})

        session = {
            "resume_runtime_overrides": {
                "reasoning_config_override": {"enabled": True, "effort": "xhigh"},
            }
        }
        _sync_create_reasoning_override_from_resume_overrides(session)
        import yaml

        cfg = yaml.safe_load(get_config_path().read_text())
        assert cfg["agent"]["reasoning_effort"] == "medium"


class TestBranchModelConfigMerge:
    def test_later_reasoning_config_merges_with_branched_from(self, tmp_path):
        db_path = tmp_path / "state.db"
        db = SessionDB(db_path)
        db.create_session(
            "branch-child",
            source="desktop",
            model_config={"_branched_from": "parent-id"},
        )
        db.create_session(
            "branch-child",
            source="desktop",
            model="glm-5.1",
            model_config={
                "model": "glm-5.1",
                "provider": "ollama-cloud",
                "reasoning_config": {"enabled": True, "effort": "low"},
            },
        )
        row = db.get_session("branch-child")
        cfg = json.loads(row["model_config"])
        assert cfg["_branched_from"] == "parent-id"
        assert cfg["reasoning_config"] == {"enabled": True, "effort": "low"}
