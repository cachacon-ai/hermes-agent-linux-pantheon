"""GPT-Live HTTP surface: readiness that never leaks a key, and a session exchange
that fails closed.

Two boundaries are pinned here.

*Secrets.* The desktop client is told which engine to mount from ``/status`` and
then exchanges an SDP offer for a Live answer. Neither response may carry key
material — not the value, not a prefix, not a length. ``/status`` reports whether a
key resolved, never the key itself.

*Profiles.* One backend serves every profile, so both routes take ``?profile=``.
``_run_config_scoped`` must reach the worker with the requested profile's home
(contextvars included). ``run_in_executor`` would drop that and resolve the LAUNCH
profile's config and OpenAI key — charging one tenant's voice minutes to another's
account, invisibly in single-profile use. That is what ``test_profile_scope_reaches_
the_worker_thread`` pins.

No test here contacts OpenAI: the vendor call is always mocked.
"""

import io
import json
import urllib.error

import pytest

FAKE_KEY = "sk-voice-live-must-never-appear-abcdef123456"


@pytest.fixture
def auth_client(monkeypatch, _isolate_hermes_home):
    try:
        from starlette.testclient import TestClient
    except ImportError:
        pytest.skip("fastapi/starlette not installed")

    import hermes_state
    from hermes_constants import get_hermes_home
    from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN

    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", get_hermes_home() / "state.db")
    client = TestClient(app, raise_server_exceptions=False)
    client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
    return client


@pytest.fixture
def no_vendor(monkeypatch):
    """Every test must not reach the vendor; individual tests swap this in."""
    from tools import voice_live

    def _explode(*args, **kwargs):
        raise AssertionError("the vendor must not be called")

    monkeypatch.setattr(voice_live.urllib.request, "urlopen", _explode)


class _Resp:
    def __init__(self, body: bytes):
        self._body = body

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _install_credentials(monkeypatch, api_key=FAKE_KEY):
    from tools import voice_live

    monkeypatch.setattr(
        voice_live,
        "_resolve_credentials",
        lambda live: (api_key, "https://api.example/v1"),
    )


def _install_voice(monkeypatch, **voice):
    from tools import voice_live

    monkeypatch.setattr(voice_live, "_voice_section", lambda: dict(voice))
    monkeypatch.setattr(voice_live, "_live_section", lambda voice=None: voice or {})


def _assert_no_key(value, seen=None):
    """Recursively assert no key material anywhere in a response payload."""
    if isinstance(value, dict):
        for item in value.values():
            _assert_no_key(item, seen)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _assert_no_key(item, seen)
    elif isinstance(value, str):
        assert FAKE_KEY not in value
        assert "sk-voice-live" not in value


class TestStatus:
    def test_defaults_to_chained_and_unconfigured(self, auth_client, monkeypatch, no_vendor):
        """A fresh install must report the chained engine and refuse to pretend GPT-Live is
        ready — the port is inert until an operator opts in."""
        _install_voice(monkeypatch)
        _install_credentials(monkeypatch, api_key="")

        body = auth_client.get("/api/audio/voice-live/status").json()

        assert body["ok"] is True
        assert body["mode"] == "chained"
        assert body["configured"] is False

    def test_reports_configured_when_the_mode_and_key_resolve(self, auth_client, monkeypatch, no_vendor):
        _install_voice(
            monkeypatch,
            voice_chat_mode="gpt-live",
            gpt_live={"model": "gpt-live-1", "voice": "marin", "instructions": ""},
        )
        _install_credentials(monkeypatch)

        body = auth_client.get("/api/audio/voice-live/status").json()

        assert body["mode"] == "gpt-live"
        assert body["configured"] is True
        assert body["model"] == "gpt-live-1"
        assert body["voice"] == "marin"
        _assert_no_key(body)

    def test_status_never_carries_key_material(self, auth_client, monkeypatch, no_vendor):
        """Every field, including any future one, is checked: a key has no business in a
        readiness answer."""
        _install_voice(monkeypatch, voice_chat_mode="gpt-live")
        _install_credentials(monkeypatch)

        raw = auth_client.get("/api/audio/voice-live/status").text

        assert FAKE_KEY not in raw
        assert "available" not in raw  # not an entitlement check

    def test_profile_scope_reaches_the_worker_thread(self, auth_client, tmp_path):
        """``asyncio.to_thread`` copies the request context; ``run_in_executor`` does not.
        With the wrong one, a secondary-profile request resolves the launch profile's
        config and key."""
        import asyncio

        from hermes_constants import (
            get_hermes_home,
            reset_hermes_home_override,
            set_hermes_home_override,
        )
        from hermes_cli.web_server import _run_config_scoped

        other_home = tmp_path / "profile-b"
        other_home.mkdir()
        token = set_hermes_home_override(other_home)
        try:
            seen = asyncio.run(_run_config_scoped("current", get_hermes_home))
        finally:
            reset_hermes_home_override(token)

        assert seen == other_home


class TestSessionExchange:
    OFFER = "v=0\r\no=- 1 1 IN IP4 127.0.0.1\r\ns=-\r\nt=0 0\r\na=group:BUNDLE 0\r\n"

    def _post(self, client, sdp, **kwargs):
        return client.post("/api/audio/voice-live/session", json={"sdp": sdp, **kwargs})

    def test_empty_sdp_is_refused_before_anything_else(self, auth_client, no_vendor):
        for sdp in ("", "   ", "\r\n"):
            resp = self._post(auth_client, sdp)

            assert resp.status_code == 400, sdp
            assert resp.json()["detail"] == "An SDP offer is required"

    def test_missing_sdp_field_is_a_validation_error(self, auth_client, no_vendor):
        resp = auth_client.post("/api/audio/voice-live/session", json={})

        assert resp.status_code == 422

    def test_no_key_is_503_and_the_vendor_is_never_called(self, auth_client, monkeypatch):
        _install_voice(monkeypatch, voice_chat_mode="gpt-live")
        _install_credentials(monkeypatch, api_key="")
        from tools import voice_live

        monkeypatch.setattr(
            voice_live.urllib.request,
            "urlopen",
            lambda *a, **k: pytest.fail("must not call the vendor"),
        )

        resp = self._post(auth_client, self.OFFER)

        assert resp.status_code == 503
        assert "API key" in resp.json()["detail"]

    def test_successful_exchange_returns_the_vendor_answer(self, auth_client, monkeypatch):
        _install_voice(monkeypatch, voice_chat_mode="gpt-live")
        _install_credentials(monkeypatch)
        from tools import voice_live

        answer = {"session": {"id": "live_x"}, "transport": {"type": "webrtc", "sdp": "v=0\r\n"}}
        monkeypatch.setattr(
            voice_live.urllib.request,
            "urlopen",
            lambda *a, **k: _Resp(json.dumps(answer).encode()),
        )

        resp = self._post(auth_client, self.OFFER)

        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is True
        assert body["transport"]["sdp"] == "v=0\r\n"
        _assert_no_key(body)

    def test_sdp_is_forwarded_byte_exact_including_the_trailing_crlf(self, auth_client, monkeypatch):
        """A stripped offer answers 400 "failed to unmarshal SDP: EOF". The route validates
        emptiness but must not reshape what it forwards."""
        _install_voice(monkeypatch, voice_chat_mode="gpt-live")
        _install_credentials(monkeypatch)
        from tools import voice_live

        captured = {}

        def fake_urlopen(req, timeout=0):
            captured["sdp"] = json.loads(req.data)["transport"]["sdp"]
            return _Resp(b"{}")

        monkeypatch.setattr(voice_live.urllib.request, "urlopen", fake_urlopen)

        resp = self._post(auth_client, self.OFFER)

        assert resp.status_code == 200
        assert captured["sdp"] == self.OFFER

    @pytest.mark.parametrize("status", [401, 403])
    def test_vendor_auth_failure_becomes_502_without_leaking_the_key(
        self, auth_client, monkeypatch, status
    ):
        _install_voice(monkeypatch, voice_chat_mode="gpt-live")
        _install_credentials(monkeypatch)
        from tools import voice_live

        def raise_http_error(req, timeout=0):
            raise urllib.error.HTTPError(
                req.full_url, status, "Unauthorized", {}, io.BytesIO(b'{"error":"bad key"}')
            )

        monkeypatch.setattr(voice_live.urllib.request, "urlopen", raise_http_error)

        resp = self._post(auth_client, self.OFFER)

        assert resp.status_code == 502
        assert FAKE_KEY not in resp.text
        _assert_no_key(resp.json())

    def test_vendor_timeout_becomes_502(self, auth_client, monkeypatch):
        _install_voice(monkeypatch, voice_chat_mode="gpt-live")
        _install_credentials(monkeypatch)
        from tools import voice_live

        def raise_timeout(req, timeout=0):
            raise urllib.error.URLError(TimeoutError("timed out"))

        monkeypatch.setattr(voice_live.urllib.request, "urlopen", raise_timeout)

        resp = self._post(auth_client, self.OFFER)

        assert resp.status_code == 502
        assert FAKE_KEY not in resp.text

    def test_vendor_network_error_becomes_502(self, auth_client, monkeypatch):
        _install_voice(monkeypatch, voice_chat_mode="gpt-live")
        _install_credentials(monkeypatch)
        from tools import voice_live

        def raise_network(req, timeout=0):
            raise urllib.error.URLError("connection reset")

        monkeypatch.setattr(voice_live.urllib.request, "urlopen", raise_network)

        assert self._post(auth_client, self.OFFER).status_code == 502

    def test_malformed_vendor_answer_becomes_502(self, auth_client, monkeypatch):
        """A 200 whose body is not JSON is still a failed session."""
        _install_voice(monkeypatch, voice_chat_mode="gpt-live")
        _install_credentials(monkeypatch)
        from tools import voice_live

        monkeypatch.setattr(
            voice_live.urllib.request, "urlopen", lambda *a, **k: _Resp(b"<html>not json</html>")
        )

        resp = self._post(auth_client, self.OFFER)

        assert resp.status_code == 502
        assert FAKE_KEY not in resp.text