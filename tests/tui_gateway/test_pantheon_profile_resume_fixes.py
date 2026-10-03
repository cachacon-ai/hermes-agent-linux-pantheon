"""Pantheon new-bot plan: Hermes-side resume/profile/session fixes."""

from __future__ import annotations

import importlib
import json
import os
import sys
import types
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from hermes_cli.config import get_config_path, save_config
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


@pytest.fixture()
def resume_server(hermes_home, monkeypatch):
    server = _import_tui_server()
    import hermes_state

    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", hermes_home / "state.db")
    monkeypatch.setattr(server, "_db", None, raising=False)
    monkeypatch.setattr(server, "_db_error", None, raising=False)
    monkeypatch.setattr(server, "_hermes_home", str(hermes_home), raising=False)
    monkeypatch.setattr(server, "_find_live_session_by_key", lambda *_a, **_k: None)
    monkeypatch.setattr(server, "_enable_gateway_prompts", lambda: None)
    monkeypatch.setattr(
        server,
        "_make_agent",
        lambda *a, **k: types.SimpleNamespace(model="test-model", close=lambda: None),
    )
    save_config({"model": {"provider": "openrouter", "default": "test-model"}, "agent": {"reasoning_effort": "medium"}})
    known = set(server._sessions)
    yield server
    live = [
        sid
        for sid in list(server._sessions)
        if sid not in known
    ]
    for sid in live:
        server.handle_request(
            {"id": "close", "method": "session.close", "params": {"session_id": sid}}
        )
    with server._sessions_lock:
        for sid in live:
            server._sessions.pop(sid, None)


def _seed_resume_row(db: SessionDB, *, title: str, model_config: dict) -> str:
    sid = uuid.uuid4().hex[:12]
    db.create_session(
        sid,
        source="desktop",
        model=str(model_config.get("model") or "test-model"),
        model_config=model_config,
    )
    db.set_session_title(sid, title)
    db.append_message(sid, "user", "hello")
    db.append_message(sid, "assistant", "hi")
    return sid


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
    def test_apply_main_model_assignment_clears_key_env_not_inline_api_key(self):
        with patch.dict(
            sys.modules,
            {"hermes_cli.env_loader": MagicMock(), "hermes_cli.banner": MagicMock()},
        ):
            from hermes_cli.web_server import _apply_main_model_assignment

        model_cfg = {
            "provider": "engramhalo",
            "default": "qwen3.8-flash-next",
            "key_env": "ENGRAMHALO_API_KEY",
            "api_key": "sk-inline-should-stay",
        }
        _apply_main_model_assignment(model_cfg, "ollama-cloud", "glm-5.1")
        assert model_cfg["provider"] == "ollama-cloud"
        assert "key_env" not in model_cfg
        assert model_cfg["api_key"] == "sk-inline-should-stay"


class TestEagerResumeReasoningOnPublishedSession:
    @pytest.mark.parametrize(
        "title",
        ["Regular chat", "Bot Chat"],
    )
    def test_eager_resume_sets_create_reasoning_on_live_session(
        self, resume_server, hermes_home, title
    ):
        db = SessionDB(db_path=hermes_home / "state.db")
        reasoning = {"enabled": True, "effort": "xhigh" if title == "Bot Chat" else "low"}
        model_config = {
            "model": "test-model",
            "provider": "openrouter",
            "reasoning_config": reasoning,
        }
        if title == "Bot Chat":
            model_config["follow_profile_config"] = True
        sid = _seed_resume_row(db, title=title, model_config=model_config)
        db.close()

        effort_before = yaml.safe_load(get_config_path().read_text())["agent"]["reasoning_effort"]

        resp = resume_server.handle_request(
            {
                "id": "resume",
                "method": "session.resume",
                "params": {"session_id": sid, "eager_build": True, "omit_messages": True},
            }
        )
        assert "error" not in resp, resp.get("error")
        live_sid = resp["result"]["session_id"]
        published = resume_server._sessions[live_sid]
        assert published["create_reasoning_override"] == reasoning

        effort_after = yaml.safe_load(get_config_path().read_text())["agent"]["reasoning_effort"]
        assert effort_after == effort_before == "medium"


class TestBranchModelConfigMerge:
    def test_later_reasoning_config_merges_with_branched_from(self, tmp_path):
        db = SessionDB(tmp_path / "state.db")
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
        db.create_session(
            "branch-child",
            source="desktop",
            model_config={"reasoning_config": None, "max_tokens": None},
        )
        row = db.get_session("branch-child")
        cfg = json.loads(row["model_config"])
        assert cfg["_branched_from"] == "parent-id"
        assert cfg["reasoning_config"] == {"enabled": True, "effort": "low"}
