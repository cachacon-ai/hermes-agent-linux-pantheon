#!/usr/bin/env python3
"""Read-only GPT-Live auth probe for BeyondMax (Chris personal use).

Exercises ChatGPT/Codex subscription quicksilver WebRTC call creation and,
optionally, the #132588 hypothesis (Codex bearer on OpenAI /v1/live/sessions).

Does not modify Hermes config, restart services, or write auth.json unless
--write-refreshed-tokens is passed after an in-memory OAuth refresh.

Usage:
  python3 scripts/probe_gpt_live_subscription_auth.py
  python3 scripts/probe_gpt_live_subscription_auth.py --variant api-live-sessions
  HERMES_HOME=~/.hermes python3 scripts/probe_gpt_live_subscription_auth.py --dry-run
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

CODEX_OAUTH_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
CODEX_OAUTH_TOKEN_URL = "https://auth.openai.com/oauth/token"
REFRESH_SKEW_SECONDS = 120

SUBSCRIPTION_URL = (
    "https://chatgpt.com/backend-api/codex/realtime/calls"
    "?intent=quicksilver&architecture=avas"
)
API_LIVE_SESSIONS_URL = "https://api.openai.com/v1/live/sessions"

# Minimal WebRTC offer with an audio m-line (hermes-talk / vendor validators).
MINIMAL_SDP_OFFER = (
    "v=0\r\n"
    "o=- 0 0 IN IP4 127.0.0.1\r\n"
    "s=probe\r\n"
    "t=0 0\r\n"
    "m=audio 9 UDP/TLS/RTP/SAVPF 0\r\n"
    "c=IN IP4 0.0.0.0\r\n"
    "a=rtcp:9 IN IP4 0.0.0.0\r\n"
)

CALL_ID_RE = re.compile(
    r"^(?:rtc_[A-Za-z0-9_-]+|"
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12})$"
)


def _hermes_home() -> Path:
    raw = os.environ.get("HERMES_HOME", "").strip()
    if raw:
        return Path(raw).expanduser()
    return Path.home() / ".hermes"


def _decode_jwt_claims(token: str) -> dict[str, Any]:
    if token.count(".") != 2:
        return {}
    payload = token.split(".")[1]
    payload += "=" * ((4 - len(payload) % 4) % 4)
    try:
        raw = base64.urlsafe_b64decode(payload.encode("utf-8"))
        data = json.loads(raw.decode("utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _token_expiring(access_token: str, skew_seconds: int) -> bool:
    claims = _decode_jwt_claims(access_token)
    exp = claims.get("exp")
    if not isinstance(exp, (int, float)):
        return False
    return float(exp) <= (time.time() + max(0, skew_seconds))


def _chatgpt_account_id(access_token: str) -> str | None:
    auth = _decode_jwt_claims(access_token).get("https://api.openai.com/auth", {})
    if not isinstance(auth, dict):
        return None
    acct = auth.get("chatgpt_account_id")
    return acct.strip() if isinstance(acct, str) and acct.strip() else None


def _redact(value: str, keep: int = 6) -> str:
    value = value or ""
    if len(value) <= keep * 2:
        return "***"
    return f"{value[:keep]}…{value[-keep:]}"


def _load_codex_tokens(auth_path: Path) -> dict[str, str]:
    if not auth_path.is_file():
        raise SystemExit(f"auth store not found: {auth_path}")
    store = json.loads(auth_path.read_text(encoding="utf-8"))
    providers = store.get("providers") if isinstance(store, dict) else None
    if not isinstance(providers, dict):
        raise SystemExit("auth.json has no providers object")
    state = providers.get("openai-codex")
    if not isinstance(state, dict):
        raise SystemExit("no providers.openai-codex in auth.json — run `hermes auth` (OpenAI Codex)")
    tokens = state.get("tokens")
    if not isinstance(tokens, dict):
        raise SystemExit("openai-codex.tokens missing or invalid")
    access = str(tokens.get("access_token") or "").strip()
    refresh = str(tokens.get("refresh_token") or "").strip()
    if not access or not refresh:
        raise SystemExit("Codex access_token or refresh_token empty")
    return {"access_token": access, "refresh_token": refresh}


def _refresh_tokens(refresh_token: str) -> dict[str, Any]:
    body = urllib.parse.urlencode(
        {
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": CODEX_OAUTH_CLIENT_ID,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        CODEX_OAUTH_TOKEN_URL,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
            "User-Agent": "hermes-gpt-live-probe/1.0",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:500]
        raise SystemExit(f"OAuth refresh failed HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise SystemExit(f"OAuth refresh network error: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit("OAuth refresh returned non-object JSON")
    return payload


def _maybe_write_tokens(auth_path: Path, tokens: dict[str, str], write: bool) -> None:
    if not write:
        return
    store = json.loads(auth_path.read_text(encoding="utf-8"))
    providers = store.setdefault("providers", {})
    state = providers.setdefault("openai-codex", {})
    state["tokens"] = {
        "access_token": tokens["access_token"],
        "refresh_token": tokens.get("refresh_token") or state.get("tokens", {}).get("refresh_token"),
    }
    state["last_refresh"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    auth_path.write_text(json.dumps(store, indent=2) + "\n", encoding="utf-8")
    print("(wrote refreshed tokens to auth.json)")


def _resolve_bearer(
    auth_path: Path,
    *,
    write_refreshed: bool,
) -> tuple[str, str, bool]:
    tokens = _load_codex_tokens(auth_path)
    refreshed = False
    access = tokens["access_token"]
    refresh = tokens["refresh_token"]
    if _token_expiring(access, REFRESH_SKEW_SECONDS):
        print("access_token within refresh skew; refreshing in memory…")
        new = _refresh_tokens(refresh)
        access = str(new.get("access_token") or "").strip()
        if not access:
            raise SystemExit("refresh response missing access_token")
        if isinstance(new.get("refresh_token"), str) and new["refresh_token"].strip():
            refresh = new["refresh_token"].strip()
        tokens = {"access_token": access, "refresh_token": refresh}
        refreshed = True
        _maybe_write_tokens(auth_path, tokens, write_refreshed)
    account = _chatgpt_account_id(access)
    if not account:
        raise SystemExit("JWT missing chatgpt_account_id claim")
    return access, account, refreshed


def _http_request(
    url: str,
    *,
    method: str,
    headers: dict[str, str],
    body: bytes | None,
    timeout: float = 45.0,
) -> tuple[int, dict[str, str], bytes]:
    req = urllib.request.Request(url, data=body, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status = int(getattr(resp, "status", 200) or 200)
            hdrs = {k.lower(): v for k, v in resp.headers.items()}
            return status, hdrs, resp.read()
    except urllib.error.HTTPError as exc:
        hdrs = {k.lower(): v for k, v in exc.headers.items()} if exc.headers else {}
        return int(exc.code), hdrs, exc.read()


def _summarize_body(body: bytes, content_type: str) -> str:
    text = body.decode("utf-8", "replace")
    if "json" in content_type:
        try:
            return json.dumps(json.loads(text), indent=2)[:2000]
        except json.JSONDecodeError:
            pass
    if text.startswith("v=0"):
        return f"<SDP {len(text)} bytes, starts with v=0>"
    return text[:2000]


def _extract_call_id(status: int, headers: dict[str, str], body: bytes) -> str | None:
    location = headers.get("location", "")
    if location:
        path = urllib.parse.urlparse(location).path
        for part in reversed([p for p in path.split("/") if p]):
            if CALL_ID_RE.fullmatch(part):
                return part
    openai_session = headers.get("openai-session-id", "")
    if CALL_ID_RE.fullmatch(openai_session):
        return openai_session
    if status in {200, 201} and body.decode("utf-8", "replace").startswith("v=0"):
        # Subscription answers are SDP-only; id must come from headers.
        return None
    try:
        payload = json.loads(body.decode("utf-8"))
        session = payload.get("session") if isinstance(payload, dict) else None
        if isinstance(session, dict):
            sid = session.get("id")
            if isinstance(sid, str) and CALL_ID_RE.fullmatch(sid):
                return sid
    except json.JSONDecodeError:
        pass
    return None


def _subscription_session_config() -> dict[str, Any]:
    return {
        "model": "gpt-live-1-codex",
        "instructions": "Probe session only. Do not speak.",
        "audio": {"output": {"voice": "cove"}},
        "delegation": {"type": "client"},
    }


def _probe_subscription(access: str, account: str, dry_run: bool) -> None:
    print("\n=== Variant: subscription (quicksilver) ===")
    print(f"URL: {SUBSCRIPTION_URL}")
    headers = {
        "Authorization": f"Bearer {_redact(access)}",
        "ChatGPT-Account-Id": _redact(account),
        "OpenAI-Alpha": "quicksilver=v2",
        "Content-Type": "application/json",
        "Accept": "*/*",
        "User-Agent": "hermes-gpt-live-probe/1.0",
    }
    payload = {"sdp": MINIMAL_SDP_OFFER, "session": _subscription_session_config()}
    print("request headers (redacted):", json.dumps(headers, indent=2))
    if dry_run:
        print("dry-run: skipping HTTP")
        return
    real_headers = {
        "Authorization": f"Bearer {access}",
        "ChatGPT-Account-Id": account,
        "OpenAI-Alpha": "quicksilver=v2",
        "Content-Type": "application/json",
        "Accept": "*/*",
        "User-Agent": "hermes-gpt-live-probe/1.0",
    }
    status, resp_headers, body = _http_request(
        SUBSCRIPTION_URL,
        method="POST",
        headers=real_headers,
        body=json.dumps(payload).encode("utf-8"),
    )
    ctype = resp_headers.get("content-type", "")
    print(f"response: HTTP {status}")
    print("response headers:", json.dumps(resp_headers, indent=2))
    print("response body summary:", _summarize_body(body, ctype))
    call_id = _extract_call_id(status, resp_headers, body)
    if call_id:
        print(f"call_id: {call_id}")
        _attempt_hangup_subscription(access, account, call_id)
    elif status in {200, 201}:
        print("call_id: not found in headers (auth may still have succeeded)")


def _probe_api_live_sessions(access: str, dry_run: bool) -> None:
    print("\n=== Variant: api-live-sessions (#132588 claim) ===")
    print(f"URL: {API_LIVE_SESSIONS_URL}")
    session = {
        "model": "gpt-live-1",
        "instructions": "Probe only.",
        "audio": {"output": {"voice": "marin"}},
        "delegation": {"type": "client"},
    }
    payload = {
        "session": session,
        "transport": {"type": "webrtc", "sdp": MINIMAL_SDP_OFFER},
    }
    if dry_run:
        print("dry-run: skipping HTTP")
        return
    headers = {
        "Authorization": f"Bearer {access}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "hermes-gpt-live-probe/1.0",
    }
    status, resp_headers, body = _http_request(
        API_LIVE_SESSIONS_URL,
        method="POST",
        headers=headers,
        body=json.dumps(payload).encode("utf-8"),
    )
    ctype = resp_headers.get("content-type", "")
    print(f"response: HTTP {status}")
    print("response headers:", json.dumps(resp_headers, indent=2))
    print("response body summary:", _summarize_body(body, ctype))
    call_id = _extract_call_id(status, resp_headers, body)
    if call_id:
        print(f"session_id: {call_id}")
        _attempt_hangup_api(access, call_id)


def _attempt_hangup_subscription(access: str, account: str, call_id: str) -> None:
    candidates = [
        ("DELETE", f"https://chatgpt.com/backend-api/codex/realtime/calls/{call_id}"),
        ("POST", f"https://chatgpt.com/backend-api/codex/realtime/calls/{call_id}/hangup"),
    ]
    for method, url in candidates:
        print(f"hangup attempt: {method} {url}")
        status, hdrs, body = _http_request(
            url,
            method=method,
            headers={
                "Authorization": f"Bearer {access}",
                "ChatGPT-Account-Id": account,
                "OpenAI-Alpha": "quicksilver=v2",
                "User-Agent": "hermes-gpt-live-probe/1.0",
            },
            body=b"" if method == "POST" else None,
        )
        print(f"  -> HTTP {status}, body: {_summarize_body(body, hdrs.get('content-type', ''))}")
        if status in {200, 201, 204, 404, 410}:
            break


def _attempt_hangup_api(access: str, session_id: str) -> None:
    url = f"{API_LIVE_SESSIONS_URL}/{urllib.parse.quote(session_id, safe='')}/hangup"
    print(f"hangup attempt: POST {url}")
    status, hdrs, body = _http_request(
        url,
        method="POST",
        headers={
            "Authorization": f"Bearer {access}",
            "Accept": "*/*",
            "User-Agent": "hermes-gpt-live-probe/1.0",
        },
        body=b"",
    )
    print(f"  -> HTTP {status}, body: {_summarize_body(body, hdrs.get('content-type', ''))}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        choices=("subscription", "api-live-sessions", "both"),
        default="subscription",
        help="subscription=quicksilver (default); api-live-sessions=#132588 check; both=run both",
    )
    parser.add_argument(
        "--write-refreshed-tokens",
        action="store_true",
        help="Persist OAuth refresh to auth.json (default: in-memory only)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Resolve tokens only; no vendor HTTP")
    args = parser.parse_args()

    home = _hermes_home()
    auth_path = home / "auth.json"
    print(f"HERMES_HOME: {home}")
    print(f"auth.json: {auth_path}")

    access, account, refreshed = _resolve_bearer(
        auth_path, write_refreshed=args.write_refreshed_tokens
    )
    exp = _decode_jwt_claims(access).get("exp")
    print(f"access_token: {_redact(access)} refreshed={refreshed} exp={exp}")
    print(f"chatgpt_account_id: {_redact(account)}")

    if args.variant in {"subscription", "both"}:
        _probe_subscription(access, account, args.dry_run)
    if args.variant in {"api-live-sessions", "both"}:
        _probe_api_live_sessions(access, args.dry_run)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
