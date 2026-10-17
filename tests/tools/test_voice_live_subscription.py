"""GPT-Live subscription billing uses Codex OAuth and never falls back to API keys."""

import base64
import json
import time
import urllib.error

import httpx
import pytest

from tools import voice_live


def _token(account="account-fixture", *, exp_offset: int = 3600):
    claims = {
        "exp": int(time.time()) + exp_offset,
        "https://api.openai.com/auth": {"chatgpt_account_id": account},
    }
    payload = base64.urlsafe_b64encode(json.dumps(claims).encode()).decode().rstrip("=")
    return f"fixture.{payload}.fixture"


def _configure(monkeypatch, tmp_path, **live):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        json.dumps({"voice": {"voice_chat_mode": "gpt-live", "gpt_live": live}}),
        encoding="utf-8",
    )


def test_subscription_request_uses_hermes_oauth_and_its_own_wire(monkeypatch, tmp_path):
    _configure(
        monkeypatch,
        tmp_path,
        auth="subscription",
        model="api-model",
        voice="marin",
        subscription_model="gpt-live-1-codex",
        subscription_voice="cove",
    )
    token = _token()
    refresh = []
    requests = []

    def credentials(**kwargs):
        refresh.append(kwargs["refresh_if_expiring"])
        return {"api_key": token, "auth_mode": "chatgpt"}

    def on_request(request):
        requests.append(request)
        return httpx.Response(
            201,
            text="v=0\r\nanswer",
            headers={
                "Location": "/realtime/calls/rtc_fixture",
                "Content-Type": "application/sdp",
            },
        )

    import hermes_cli.auth as auth_mod

    monkeypatch.setattr(auth_mod, "resolve_codex_runtime_credentials", credentials)
    _RealClient = httpx.Client

    class _StubClient:
        def __init__(self, **kwargs):
            self._client = _RealClient(
                transport=httpx.MockTransport(on_request),
                timeout=kwargs.get("timeout", 30),
                follow_redirects=kwargs.get("follow_redirects", False),
            )

        def __enter__(self):
            return self._client.__enter__()

        def __exit__(self, *exc):
            return self._client.__exit__(*exc)

    monkeypatch.setattr(voice_live.httpx, "Client", _StubClient)
    monkeypatch.setattr(
        voice_live, "_resolve_credentials", lambda *_: pytest.fail("paid API resolver")
    )
    status = voice_live.resolve_gpt_live_status()
    result = voice_live.create_webrtc_session("v=0 offer", history=[])

    assert status["auth"] == result["auth"] == "subscription"
    assert status["available"] and refresh == [False, True]
    request, = requests
    assert str(request.url) == (
        "https://chatgpt.com/backend-api/codex/realtime/calls"
        "?intent=quicksilver&architecture=avas"
    )
    assert request.headers["Authorization"] == f"Bearer {token}"
    assert request.headers["ChatGPT-Account-Id"] == "account-fixture"
    assert request.headers["OpenAI-Alpha"] == "quicksilver=v2"
    payload = json.loads(request.content)
    assert payload["sdp"] == "v=0 offer" and "transport" not in payload
    assert payload["session"]["model"] == status["model"] == "gpt-live-1-codex"
    assert payload["session"]["audio"]["output"]["voice"] == status["voice"] == "cove"
    assert result["session"]["id"] == "rtc_fixture"
    assert token not in json.dumps([status, result])


def test_status_is_read_only_and_session_auth_is_pinned(monkeypatch, tmp_path):
    _configure(monkeypatch, tmp_path, auth="subscription")
    calls = []

    def credentials(**kwargs):
        calls.append(kwargs)
        return {"api_key": _token(), "auth_mode": "chatgpt"}

    import hermes_cli.auth as auth_mod

    monkeypatch.setattr(auth_mod, "resolve_codex_runtime_credentials", credentials)
    status = voice_live.resolve_gpt_live_status()

    assert status["available"] is True
    assert calls == [{"refresh_if_expiring": False, "read_only": True}]

    with pytest.raises(voice_live.VoiceLiveAuthMismatchError):
        voice_live.create_webrtc_session("v=0 offer", expected_auth="api")

    assert len(calls) == 1


def test_subscription_session_history_uses_initial_items(monkeypatch, tmp_path):
    _configure(monkeypatch, tmp_path, auth="subscription")
    token = _token()
    requests = []

    def on_request(request):
        requests.append(request)
        return httpx.Response(
            201,
            text="v=0\r\nanswer",
            headers={"Location": "/realtime/calls/rtc_hist", "Content-Type": "application/sdp"},
        )

    import hermes_cli.auth as auth_mod

    monkeypatch.setattr(
        auth_mod, "resolve_codex_runtime_credentials", lambda **kw: {"api_key": token}
    )
    _RealClient = httpx.Client

    class _StubClient:
        def __init__(self, **kwargs):
            self._client = _RealClient(
                transport=httpx.MockTransport(on_request),
                timeout=kwargs.get("timeout", 30),
                follow_redirects=kwargs.get("follow_redirects", False),
            )

        def __enter__(self):
            return self._client.__enter__()

        def __exit__(self, *exc):
            return self._client.__exit__(*exc)

    monkeypatch.setattr(voice_live.httpx, "Client", _StubClient)
    voice_live.create_webrtc_session(
        "v=0 offer",
        history=[{"role": "user", "content": "hello from history"}],
    )
    payload = json.loads(requests[0].content)
    assert "input" not in payload["session"]
    assert payload["session"]["initial_items"][0]["role"] == "user"


@pytest.mark.parametrize("status_code", [404, 410])
def test_subscription_hangup_posts_realtime_url_and_treats_gone_as_closed(
    monkeypatch, tmp_path, status_code,
):
    _configure(monkeypatch, tmp_path, auth="subscription")
    token = _token()
    captured = []

    def urlopen(req, timeout=30):
        captured.append(req)
        raise urllib.error.HTTPError(req.full_url, status_code, "", {}, None)

    import hermes_cli.auth as auth_mod

    monkeypatch.setattr(auth_mod, "resolve_codex_runtime_credentials", lambda **kw: {"api_key": token})
    monkeypatch.setattr(voice_live.urllib.request, "urlopen", urlopen)

    result = voice_live.close_live_session("rtc_close_fixture")

    assert result == {"closed": True, "method": "hangup", "already_closed": True}
    req = captured[0]
    assert req.get_full_url() == (
        "https://api.openai.com/v1/realtime/calls/rtc_close_fixture/hangup"
    )
    headers = {name.lower(): value for name, value in req.header_items()}
    assert headers["authorization"] == f"Bearer {token}"
    assert headers["chatgpt-account-id"] == "account-fixture"


def _write_auth_store(hermes_home, payload):
    hermes_home.mkdir(parents=True, exist_ok=True)
    (hermes_home / "auth.json").write_text(json.dumps(payload), encoding="utf-8")


def _pool_rate_limited_store(now=None):
    now = now or time.time()
    return {
        "version": 1,
        "providers": {},
        "credential_pool": {
            "openai-codex": [
                {
                    "id": "cred-quota",
                    "label": "quota-frozen",
                    "auth_type": "oauth",
                    "priority": 0,
                    "source": "device_code",
                    "access_token": "tok-quota",
                    "last_status": "exhausted",
                    "last_status_at": now,
                    "last_error_code": 429,
                    "last_error_reason": "usage_limit_reached",
                    "last_error_message": "The usage limit has been reached",
                    "last_error_reset_at": now + 3 * 24 * 3600,
                }
            ]
        },
    }


def test_subscription_status_pool_cooldown_skips_probe_and_auth_write(tmp_path, monkeypatch):
    _configure(monkeypatch, tmp_path, auth="subscription")
    hermes_home = tmp_path
    _write_auth_store(hermes_home, _pool_rate_limited_store())
    import hermes_cli.auth as auth_mod

    monkeypatch.setattr(
        auth_mod,
        "_probe_codex_pool_entry_quota_restored",
        lambda *a, **k: pytest.fail("quota probe must not run on read-only status"),
    )
    monkeypatch.setattr(
        auth_mod,
        "clear_codex_pool_quota_cooldowns",
        lambda *a, **k: pytest.fail("cooldown clear must not run on read-only status"),
    )
    monkeypatch.setattr(
        auth_mod,
        "_save_auth_store",
        lambda *a, **k: pytest.fail("auth store must not be written on read-only status"),
    )

    before = json.loads((hermes_home / "auth.json").read_text())
    status = voice_live.resolve_gpt_live_status()
    after = json.loads((hermes_home / "auth.json").read_text())

    assert status["auth"] == "subscription"
    assert status["available"] is False
    assert status["reason"]
    assert "quota" in status["reason"].lower() or "429" in status["reason"]
    assert after == before


def test_subscription_status_expired_token_unavailable_without_refresh(monkeypatch, tmp_path):
    _configure(monkeypatch, tmp_path, auth="subscription")
    token = _token(exp_offset=-120)
    calls = []

    def credentials(**kwargs):
        calls.append(kwargs)
        return {"api_key": token, "auth_mode": "chatgpt"}

    import hermes_cli.auth as auth_mod

    monkeypatch.setattr(auth_mod, "resolve_codex_runtime_credentials", credentials)
    status = voice_live.resolve_gpt_live_status()

    assert calls == [{"refresh_if_expiring": False, "read_only": True}]
    assert status["available"] is False
    assert "expired" in status["reason"].lower()
