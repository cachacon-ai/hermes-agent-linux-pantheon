"""Custom endpoint connection test API (PAN-14 slice 6)."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from hermes_cli.config import custom_endpoint_key_env, load_config, save_config
from hermes_cli.web_server import (
    _http_request_same_host_redirect,
    _probe_custom_endpoint_connection,
    app,
)
from tools.url_safety import (
    is_allowed_provider_endpoint_test_url,
    provider_endpoint_test_log_target,
)


@pytest.mark.parametrize(
    "url,allowed",
    [
        ("http://127.0.0.1:8080/v1", True),
        ("http://192.168.1.50:11434/v1", True),
        ("http://localhost:8080/v1", True),
        ("file:///etc/passwd", False),
        ("gopher://127.0.0.1", False),
        ("http://169.254.169.254/latest/meta-data/", False),
    ],
)
def test_provider_test_url_policy(url, allowed):
    assert is_allowed_provider_endpoint_test_url(url) is allowed


def test_provider_test_url_dns_failure_is_blocked(monkeypatch):
    import socket

    def _fail(*_args, **_kwargs):
        raise socket.gaierror("no such host")

    monkeypatch.setattr(socket, "getaddrinfo", _fail)
    assert is_allowed_provider_endpoint_test_url("http://missing.example.test/v1") is False


def test_provider_endpoint_test_log_target_strips_secrets():
    target = provider_endpoint_test_log_target(
        "https://user:sekret@127.0.0.1:8080/v1?token=abc"
    )
    assert "sekret" not in target
    assert "token" not in target
    assert target == "https://127.0.0.1:8080"


@pytest.mark.asyncio
async def test_redirect_does_not_follow_scheme_downgrade_with_auth():
    calls: list[tuple[str, dict]] = []

    class _Response:
        def __init__(self, status_code, url, location=None):
            self.status_code = status_code
            self.url = url
            self.headers = {"location": location} if location else {}

    class _Client:
        async def request(self, method, url, follow_redirects=False, **kwargs):
            calls.append((url, kwargs.get("headers") or {}))
            if url.startswith("https://"):
                return _Response(
                    302,
                    url,
                    location="http://127.0.0.1:8080/v1/models",
                )
            return _Response(200, url)

    result = await _http_request_same_host_redirect(
        _Client(),
        "GET",
        "https://127.0.0.1:8080/v1/models",
        headers={"Authorization": "Bearer secret"},
    )

    assert result.status_code == 302
    assert len(calls) == 1
    assert all("http://" not in c[0] for c in calls)


@pytest.mark.asyncio
async def test_probe_models_success_counts_models():
    response = SimpleNamespace(
        is_success=True,
        status_code=200,
        json=lambda: {"data": [{"id": "a"}, {"id": "b"}]},
    )

    class _Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        request = AsyncMock(return_value=response)

    with patch(
        "tools.url_safety.create_provider_endpoint_test_async_client",
        return_value=_Client(),
    ):
        result = await _probe_custom_endpoint_connection(
            "http://127.0.0.1:8080/v1",
            api_key="",
        )

    assert result["ok"] is True
    assert result["status"] == 200
    assert result["model_count"] == 2
    assert "sk-" not in json.dumps(result)


@pytest.mark.asyncio
async def test_probe_timeout_returns_error():
    import httpx

    class _Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def request(self, *args, **kwargs):
            raise httpx.TimeoutException("timed out")

    with patch(
        "tools.url_safety.create_provider_endpoint_test_async_client",
        return_value=_Client(),
    ):
        result = await _probe_custom_endpoint_connection(
            "http://127.0.0.1:8080/v1",
            api_key="probe-key-should-not-leak",
            timeout_seconds=0.01,
        )

    assert result["ok"] is False
    assert result["error"] == "Connection timed out"
    assert "probe-key" not in json.dumps(result)


@pytest.mark.asyncio
async def test_probe_chat_fallback_when_models_missing():
    models_resp = SimpleNamespace(is_success=False, status_code=404, url="http://127.0.0.1:8080/v1/models")
    chat_resp = SimpleNamespace(is_success=True, status_code=200, url="http://127.0.0.1:8080/v1/chat/completions")

    calls: list[str] = []

    class _Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def request(self, method, url, **kwargs):
            calls.append(f"{method} {url}")
            if url.endswith("/models"):
                return models_resp
            return chat_resp

    with patch(
        "tools.url_safety.create_provider_endpoint_test_async_client",
        return_value=_Client(),
    ):
        result = await _probe_custom_endpoint_connection(
            "http://127.0.0.1:8080/v1",
            api_key="",
            model="qwen",
        )

    assert result["ok"] is True
    assert any("chat/completions" in c for c in calls)


class TestCustomEndpointTestRoute:
    @pytest.fixture(autouse=True)
    def _client(self, _isolate_hermes_home):
        from fastapi.testclient import TestClient
        from hermes_cli.web_server import _SESSION_HEADER_NAME, _SESSION_TOKEN

        self.client = TestClient(app)
        self.client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN

    def test_candidate_body_does_not_persist(self, monkeypatch):
        cfg = load_config()
        cfg["providers"] = {
            "local": {
                "name": "Local",
                "base_url": "http://127.0.0.1:8080/v1",
                "model": "qwen",
            }
        }
        save_config(cfg)

        async def _fake_probe(base_url, *, api_key="", model="", timeout_seconds=8.0):
            return {
                "ok": True,
                "status": 200,
                "latency_ms": 1,
                "model_count": 1,
                "_seen_key": api_key,
            }

        monkeypatch.setattr(
            "hermes_cli.web_server._probe_custom_endpoint_connection",
            _fake_probe,
        )

        resp = self.client.post(
            "/api/providers/custom-endpoints/local/test",
            json={"base_url": "http://192.168.0.9:8080/v1", "api_key": "candidate-only"},
        )
        assert resp.status_code == 200
        assert resp.json()["ok"] is True

        cfg_after = load_config()
        assert cfg_after["providers"]["local"]["base_url"] == "http://127.0.0.1:8080/v1"
        env_var = custom_endpoint_key_env("local")
        from hermes_cli.config import get_env_value

        assert get_env_value(env_var) is None

    def test_invalid_profile_slug_rejected(self):
        resp = self.client.post(
            "/api/providers/custom-endpoints/local/test?profile=../escape",
            json={},
        )
        assert resp.status_code == 400

    def test_candidate_base_url_without_api_key_omits_stored_secret(self, monkeypatch):
        from hermes_cli.config import save_env_value

        cfg = load_config()
        cfg["providers"] = {
            "local": {
                "name": "Local",
                "base_url": "http://127.0.0.1:8080/v1",
                "model": "qwen",
                "key_env": custom_endpoint_key_env("local"),
            }
        }
        save_config(cfg)
        save_env_value(custom_endpoint_key_env("local"), "stored-profile-secret")

        seen: dict[str, str] = {}

        async def _fake_probe(base_url, *, api_key="", model="", timeout_seconds=8.0):
            seen["api_key"] = api_key
            return {"ok": True, "status": 200, "latency_ms": 1}

        monkeypatch.setattr(
            "hermes_cli.web_server._probe_custom_endpoint_connection",
            _fake_probe,
        )

        resp = self.client.post(
            "/api/providers/custom-endpoints/local/test",
            json={"base_url": "http://192.168.0.9:8080/v1"},
        )
        assert resp.status_code == 200
        assert seen.get("api_key") == ""

    def test_no_key_endpoint_test_works(self, monkeypatch):
        cfg = load_config()
        cfg["providers"] = {
            "local": {
                "name": "Local",
                "base_url": "http://127.0.0.1:8080/v1",
                "model": "qwen",
            }
        }
        save_config(cfg)

        seen: dict[str, str] = {}

        async def _fake_probe(base_url, *, api_key="", model="", timeout_seconds=8.0):
            seen["api_key"] = api_key
            return {"ok": True, "status": 200, "latency_ms": 3}

        monkeypatch.setattr(
            "hermes_cli.web_server._probe_custom_endpoint_connection",
            _fake_probe,
        )

        resp = self.client.post(
            "/api/providers/custom-endpoints/local/test",
            json={"api_key": ""},
        )
        assert resp.status_code == 200
        assert resp.json()["ok"] is True
        assert seen["api_key"] == ""
