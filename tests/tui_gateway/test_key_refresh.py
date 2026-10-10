"""Live provider credential refresh at TUI turn boundaries (PAN-14 slice 6)."""

from __future__ import annotations

from types import SimpleNamespace

from tui_gateway import server


def _agent_with_client(api_key: str = "old-key"):
    old_client = SimpleNamespace(closed=False)

    def _close():
        old_client.closed = True

    old_client.close = _close
    agent = SimpleNamespace(
        model="local-model",
        provider="my-endpoint",
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

    old_sig = (
        "my-endpoint",
        "http://127.0.0.1:8080/v1",
        ("sha256", "old-digest"),
    )
    new_sig = (
        "my-endpoint",
        "http://127.0.0.1:8080/v1",
        ("sha256", "new-digest"),
    )

    monkeypatch.setattr(
        server, "_target_provider_credential_signature", lambda _s, _a: new_sig
    )
    monkeypatch.setattr(server, "_agent_provider_credential_signature", lambda _a: old_sig)
    monkeypatch.setattr(
        server,
        "_resolve_live_agent_runtime_credentials",
        lambda _s, _a: {
            "provider": "my-endpoint",
            "base_url": "http://127.0.0.1:8080/v1",
            "api_key": "new-key-material",
            "api_mode": "chat_completions",
        },
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
    assert old_client.closed is True
    assert agent.api_key == "new-key-material"

    switch_calls.clear()
    monkeypatch.setattr(server, "_agent_provider_credential_signature", lambda _a: new_sig)
    server._sync_agent_provider_credentials_with_config("sid", session)
    assert switch_calls == []


def test_credential_refresh_never_logs_api_key(caplog, monkeypatch):
    secret = "sk-super-secret-never-log-me"
    agent, _old_client = _agent_with_client(api_key="stale")
    session = {"agent": agent, "session_key": "session-key"}

    def _boom(_session, _agent):
        raise RuntimeError(f"resolver saw {secret}")

    monkeypatch.setattr(server, "_target_provider_credential_signature", _boom)

    with caplog.at_level("WARNING"):
        server._sync_agent_provider_credentials_with_config("sid", session)

    combined = caplog.text
    assert secret not in combined


def test_gateway_capabilities_advertises_provider_reload_flags():
    caps = server._methods["gateway.capabilities"]("caps", {})["result"]
    assert caps.get("provider_key_refresh") is True
    assert caps.get("provider_test") is True
