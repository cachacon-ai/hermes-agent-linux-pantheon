"""GPT-Live subscription billing uses Codex OAuth and never falls back to API keys."""

import base64
import json

import httpx
import pytest

from hermes_cli.auth import AuthError
from tools import voice_live


def _token(account="account-fixture"):
    claims = {"https://api.openai.com/auth": {"chatgpt_account_id": account}}
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
