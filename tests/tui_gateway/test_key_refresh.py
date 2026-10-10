"""Live provider credential refresh at TUI turn boundaries (PAN-14 slice 6)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tui_gateway import server


def _agent_with_client(api_key: str = "old-key", provider: str = "my-endpoint"):
    old_client = SimpleNamespace(closed=False)

    def _close():
        old_client.closed = True

    old_client.close = _close
    agent = SimpleNamespace(
        model="local-model",
        provider=provider,
        base_url="http://127.0.0.1:8080/v1",
        api_key=api_key,
        api_mode="chat_completions",
        client=old_client,
        runtime_capabilities={},
    )
    return agent, old_client


def test_key_change_rebuilds_client_on_next_turn(monkeypatch):
    agent, old_client = _agent_with_client()
    session = {"agent": agent, "session_key": "session-key"}

    resolution = server._LiveCredentialResolution(
        {
            "provider": "my-endpoint",
            "base_url": "http://127.0.0.1:8080/v1",
            "api_key": "new-key-material",
            "api_mode": "chat_completions",
        },
        used_fallback=False,
    )
    monkeypatch.setattr(
        server,
        "_resolve_live_agent_runtime_credentials",
        lambda _s, _a: resolution,
    )
    monkeypatch.setattr(server, "_restart_slash_worker", lambda *_args: None)
    monkeypatch.setattr(server, "_persist_live_session_runtime", lambda *_args: None)

    switch_calls: list[dict] = []

    def _switch_model(**kwargs):
        switch_calls.append(kwargs)
        agent.api_key = kwargs.get("api_key")
        agent.client = SimpleNamespace(closed=False, close=lambda: None)
        old_client.close()

    agent.switch_model = _switch_model

    server._sync_agent_provider_credentials_with_config("sid", session)

    assert switch_calls
    assert switch_calls[0]["api_key"] == "new-key-material"
    assert switch_calls[0]["new_provider"] == "my-endpoint"
    assert old_client.closed is True
    assert agent.api_key == "new-key-material"

    switch_calls.clear()
    server._sync_agent_provider_credentials_with_config("sid", session)
    assert switch_calls == []


def test_credential_refresh_never_logs_api_key(caplog, monkeypatch):
    secret = "sk-super-secret-never-log-me"
    agent, _old_client = _agent_with_client(api_key="stale")
    session = {"agent": agent, "session_key": "session-key"}

    def _boom(_session, _agent):
        raise RuntimeError(f"resolver saw {secret}")

    monkeypatch.setattr(server, "_resolve_live_agent_runtime_credentials", _boom)

    with caplog.at_level("WARNING"):
        server._sync_agent_provider_credentials_with_config("sid", session)

    combined = caplog.text
    assert secret not in combined


def test_gateway_capabilities_advertises_provider_reload_flags():
    caps = server._methods["gateway.capabilities"]("caps", {})["result"]
    assert caps.get("provider_key_refresh") is True
    assert caps.get("provider_test") is True


def test_used_fallback_skips_credential_refresh(monkeypatch):
    agent, _old_client = _agent_with_client()
    session = {"agent": agent, "session_key": "session-key"}

    resolution = server._LiveCredentialResolution(
        {
            "provider": "openrouter",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key": "fallback-key",
            "api_mode": "chat_completions",
        },
        used_fallback=True,
    )
    monkeypatch.setattr(
        server,
        "_resolve_live_agent_runtime_credentials",
        lambda _s, _a: resolution,
    )

    agent.switch_model = lambda **_kw: pytest.fail("must not switch on fallback")

    server._sync_agent_provider_credentials_with_config("sid", session)


def test_empty_resolved_key_does_not_retry_every_turn(monkeypatch):
    agent, _old_client = _agent_with_client(api_key="still-here")
    session = {"agent": agent, "session_key": "session-key"}

    empty_resolution = server._LiveCredentialResolution(
        {
            "provider": "my-endpoint",
            "base_url": "http://127.0.0.1:8080/v1",
            "api_key": "",
            "api_mode": "chat_completions",
        },
        used_fallback=False,
    )
    monkeypatch.setattr(
        server,
        "_resolve_live_agent_runtime_credentials",
        lambda _s, _a: empty_resolution,
    )

    switch_calls = 0

    def _switch_model(**_kwargs):
        nonlocal switch_calls
        switch_calls += 1

    agent.switch_model = _switch_model

    server._sync_agent_provider_credentials_with_config("sid", session)
    server._sync_agent_provider_credentials_with_config("sid", session)
    assert switch_calls == 0


def test_real_resolution_path_second_turn_is_noop(monkeypatch):
    """Exercise _sync via resolve_runtime_provider, not signature monkeypatches."""
    agent, old_client = _agent_with_client(api_key="old-key")
    session = {"agent": agent, "session_key": "session-key"}

    calls = {"n": 0}

    def _resolve_runtime_provider(**_kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return {
                "provider": "my-endpoint",
                "base_url": "http://127.0.0.1:8080/v1",
                "api_key": "rotated-key",
                "api_mode": "chat_completions",
            }
        return {
            "provider": "my-endpoint",
            "base_url": "http://127.0.0.1:8080/v1",
            "api_key": "rotated-key",
            "api_mode": "chat_completions",
        }

    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        _resolve_runtime_provider,
    )
    monkeypatch.setattr(server, "_config_model_target", lambda: ("local-model", "my-endpoint"))
    monkeypatch.setattr(server, "_restart_slash_worker", lambda *_a: None)
    monkeypatch.setattr(server, "_persist_live_session_runtime", lambda *_a: None)

    switch_calls: list[str] = []

    def _switch_model(**kwargs):
        switch_calls.append(str(kwargs.get("api_key")))
        agent.api_key = kwargs.get("api_key")
        old_client.close()

    agent.switch_model = _switch_model

    server._sync_agent_provider_credentials_with_config("sid", session)
    assert switch_calls == ["rotated-key"]

    switch_calls.clear()
    server._sync_agent_provider_credentials_with_config("sid", session)
    assert switch_calls == []
