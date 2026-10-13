"""Request validation errors must not 500 on binary or multipart bodies."""

import pytest


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


def test_transcribe_multipart_returns_422_not_500(auth_client):
    files = {"file": ("clip.webm", b"\xff\xfe\x00\x01", "audio/webm")}
    resp = auth_client.post("/api/audio/transcribe", files=files)
    assert resp.status_code == 422
    body = resp.json()
    assert "detail" in body
    detail = body["detail"]
    assert isinstance(detail, list) and detail
    body_error = next((item for item in detail if item.get("loc") == ["body"]), None)
    assert body_error is not None
    assert "application/json" in body_error["msg"]
    assert "data_url" in body_error["msg"]
    assert body_error.get("input") == "<binary request body omitted>"


def test_transcribe_invalid_json_returns_422(auth_client):
    resp = auth_client.post(
        "/api/audio/transcribe",
        content=b"not-json",
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 422
    assert "detail" in resp.json()


def test_sanitize_validation_errors_unit():
    from hermes_cli.web_validation_errors import sanitize_validation_errors

    errors = [
        {
            "type": "model_attributes_type",
            "loc": ("body",),
            "msg": "bad",
            "input": b"\xff\xfe",
        }
    ]
    sanitized = sanitize_validation_errors(errors)
    assert sanitized[0]["input"] == "<binary request body omitted>"
