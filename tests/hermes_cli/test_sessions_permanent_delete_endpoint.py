"""Permanent session deletion also erases only exact staged upload paths."""

import pytest


@pytest.fixture
def client(monkeypatch, _isolate_hermes_home):
    pytest.importorskip("fastapi")
    from starlette.testclient import TestClient

    import hermes_state
    from hermes_constants import get_hermes_home
    from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN

    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", get_hermes_home() / "state.db")
    result = TestClient(app)
    result.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
    return result


def test_permanent_delete_removes_profile_staged_paths_but_not_workspace_files(client):
    from hermes_constants import get_hermes_home
    from hermes_state import SessionDB

    profile_home = get_hermes_home()
    db = SessionDB(db_path=profile_home / "state.db")
    db.create_session("pan12-owned-session", source="api")
    db.append_message("pan12-owned-session", "user", "delete this content")
    db.close()

    attachment = profile_home / "attachments" / "client-upload.txt"
    image = profile_home / "images" / "upload.png"
    workspace = profile_home / "workspace" / "keep.txt"
    for path in (attachment, image, workspace):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(path.name)

    rejected = client.request(
        "DELETE",
        "/api/sessions/pan12-owned-session/permanent?profile=default&allow_missing=true",
        json={"attachment_paths": [str(workspace)]},
    )
    assert rejected.status_code == 400, rejected.text
    db = SessionDB(db_path=profile_home / "state.db")
    assert db.get_session("pan12-owned-session") is not None
    db.close()

    response = client.request(
        "DELETE",
        "/api/sessions/pan12-owned-session/permanent?profile=default&allow_missing=true",
        json={"attachment_paths": [str(attachment), str(image)]},
    )
    assert response.status_code == 200, response.text
    assert response.json()["deleted_session_ids"] == ["pan12-owned-session"]
    assert not attachment.exists()
    assert not image.exists()
    assert workspace.read_text() == "keep.txt"

    # A retry is safe after the durable deletion marker exists and files are gone.
    retried = client.request(
        "DELETE",
        "/api/sessions/pan12-owned-session/permanent?profile=default&allow_missing=true",
        json={"attachment_paths": [str(attachment), str(image)]},
    )
    assert retried.status_code == 200, retried.text
    assert workspace.exists()


def test_permanent_delete_rejects_directory_and_cross_profile_paths(client):
    from hermes_constants import get_hermes_home

    profile_home = get_hermes_home()
    other_profile_attachment = profile_home.parent / "other-profile" / "attachments" / "keep.txt"
    other_profile_attachment.parent.mkdir(parents=True, exist_ok=True)
    other_profile_attachment.write_text("keep")
    staged_directory = profile_home / "attachments" / "nested"
    staged_directory.mkdir(parents=True, exist_ok=True)

    for path in (staged_directory, other_profile_attachment):
        response = client.request(
            "DELETE",
            "/api/sessions/not-created/permanent?profile=default&allow_missing=true",
            json={"attachment_paths": [str(path)]},
        )
        assert response.status_code == 400, response.text
    assert other_profile_attachment.read_text() == "keep"


def test_permanent_delete_remains_compatible_with_callers_that_send_no_body(client):
    response = client.delete(
        "/api/sessions/pan12-lazy-session/permanent?profile=default&allow_missing=true"
    )
    assert response.status_code == 200, response.text
    assert response.json()["deleted_session_ids"] == ["pan12-lazy-session"]
