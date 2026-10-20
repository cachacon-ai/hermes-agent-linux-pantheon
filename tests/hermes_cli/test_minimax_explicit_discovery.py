"""Saved Hermes OAuth logins survive Pantheon's explicit catalog filter."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


_DISCOVER = r'''
import json
import sys
from unittest.mock import patch
from hermes_constants import set_hermes_home_override
if sys.argv[1]:
    set_hermes_home_override(sys.argv[1])
from hermes_cli import inventory
from hermes_cli.auth import (
    clear_provider_auth, is_provider_explicitly_configured, suppress_credential_source,
)
if sys.argv[2] == "logout":
    clear_provider_auth("minimax-oauth")
elif sys.argv[2] == "suppress":
    suppress_credential_source("minimax-oauth", "oauth")

# Replace remote metadata/enrichment only. Credential loading, pool seeding,
# config/profile resolution, explicit filtering and payload shaping stay real.
with patch("socket.socket.connect", side_effect=AssertionError("unexpected network call")), \
     patch("agent.models_dev.fetch_models_dev", return_value={}), \
     patch("hermes_cli.models.get_curated_nous_model_ids", return_value=[]), \
     patch.object(inventory, "_apply_pricing"), \
     patch.object(inventory, "_apply_capabilities"), \
     patch.object(inventory, "_apply_featured"):
    configured_before_discovery = is_provider_explicitly_configured("minimax-oauth")
    payload = inventory.build_model_options_payload(
        inventory.load_picker_context(), explicit_only=True, include_unconfigured=True,
    )
    row = next(p for p in payload["providers"] if p["slug"] == "minimax-oauth")
    print(json.dumps({"configured": configured_before_discovery, "row": row}))
'''


def _store(home, state=None, *, suppressed=False, pool=None):
    home.mkdir(parents=True, exist_ok=True)
    data = {"version": 1, "active_provider": "openai-codex", "providers": {}}
    if state is not None:
        data["providers"]["minimax-oauth"] = state
    if suppressed:
        data["suppressed_sources"] = {"minimax-oauth": ["oauth"]}
    if pool is not None:
        data["credential_pool"] = {"minimax-oauth": pool}
    (home / "auth.json").write_text(json.dumps(data))


@pytest.fixture
def saved_login(tmp_path, monkeypatch):
    from hermes_cli.models import CANONICAL_PROVIDERS

    root = tmp_path / "hermes"
    root.mkdir()
    # Only MiniMax discovery participates; unrelated provider APIs must never
    # be contacted by this regression test.
    config = {
        "model": {"provider": "other", "default": "other-model"},
        "model_catalog": {"excluded_providers": [
            p.slug for p in CANONICAL_PROVIDERS if p.slug != "minimax-oauth"
        ]},
    }
    (root / "config.yaml").write_text(json.dumps(config))
    state = {
        "access_token": "test-minimax-access-token",
        "refresh_token": "test-minimax-refresh-token",
        "expires_at": "2099-01-01T00:00:00+00:00",
        "inference_base_url": "https://api.minimax.io/anthropic",
    }
    _store(root, state)
    monkeypatch.setenv("HERMES_HOME", str(root))
    return root, state, config


def _discover(root, profile=None, action=""):
    env = dict(os.environ, HERMES_HOME=str(root))
    args = [sys.executable, "-c", _DISCOVER, str(profile) if profile else "", action]
    result = subprocess.run(
        args, cwd=Path(__file__).resolve().parents[2], env=env,
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_saved_login_survives_explicit_filter_and_restart(saved_login):
    root, _, _ = saved_login
    first = _discover(root)
    assert first["configured"] is True
    assert first["row"]["authenticated"] is True
    assert first["row"]["models"]
    assert first["row"]["is_current"] is False
    # The first open seeds a credential pool; the next process must agree.
    second = _discover(root)
    assert second["row"] == first["row"]


@pytest.mark.parametrize("suppressed", [False, True])
def test_removed_or_suppressed_login_stays_hidden(saved_login, suppressed):
    root, _, _ = saved_login
    assert _discover(root)["row"]["authenticated"] is True
    result = _discover(root, action="suppress" if suppressed else "logout")
    assert result["configured"] is False
    assert result["row"]["authenticated"] is False
    assert result["row"]["models"] == []


def test_profile_inherits_login_and_respects_local_suppression(saved_login):
    root, _, config = saved_login
    profile = root / "profiles" / "tester"
    _store(profile)
    (profile / "config.yaml").write_text(json.dumps(config))
    assert _discover(root, profile)["row"]["authenticated"] is True
    assert _discover(root, profile, action="suppress")["row"]["authenticated"] is False
    assert _discover(root)["row"]["authenticated"] is True


def test_profile_provider_state_shadows_root(saved_login):
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override
    from hermes_cli.auth import get_provider_auth_state

    root, state, _ = saved_login
    profile = root / "profiles" / "tester"
    local = {**state, "access_token": "profile-token"}
    _store(profile, local)
    token = set_hermes_home_override(profile)
    try:
        assert get_provider_auth_state("minimax-oauth")["access_token"] == "profile-token"
    finally:
        reset_hermes_home_override(token)


@pytest.mark.parametrize("provider,source", [
    ("minimax-oauth", "oauth"), ("copilot", "oauth"),
    ("copilot", "gh_cli"), ("anthropic", "claude_code"),
])
def test_oauth_provenance_is_provider_specific(tmp_path, monkeypatch, provider, source):
    from hermes_cli.auth import is_provider_explicitly_configured

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text('{"model":{"provider":"other"}}')
    (tmp_path / "auth.json").write_text(json.dumps({
        "version": 1, "providers": {}, "credential_pool": {provider: [{
            "id": "test-entry", "source": source, "auth_type": "oauth",
            "access_token": "test-token",
        }]},
    }))
    assert is_provider_explicitly_configured(provider) is (provider == "minimax-oauth")
