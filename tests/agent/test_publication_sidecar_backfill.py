"""Compaction backfill targets the actual user row, never a newer neighbor."""

import pytest

from hermes_state import SessionDB


@pytest.fixture
def compacted(tmp_path):
    with SessionDB(tmp_path / "state.db") as db:
        db.create_session("session", source="pantheon")
        rows = [
            {"role": "user", "content": "Create report"},
            {"role": "assistant", "content": "Working"},
            {"role": "user", "content": "Todo snapshot", "_todo_snapshot_synthetic": True},
        ]
        db.archive_and_compact("session", rows)
        yield db, rows


def test_scoped_backfill_targets_request_before_synthetic_user(compacted):
    db, rows = compacted
    wire = "Create report\n\nPUBLICATION\n\nMEMORY\n\nPLUGIN"
    # The legacy latest-user guard correctly refuses a different newest row.
    assert db.set_latest_user_api_content("session", "Create report", wire) == 0
    assert db.set_latest_user_api_content(
        "session", "Create report", wire, row_id=rows[0]["_row_id"]
    ) == 1
    replay = db.get_messages_as_conversation("session")
    assert replay[0]["content"] == "Create report"
    assert replay[0]["api_content"] == wire
    assert "api_content" not in replay[-1]


@pytest.mark.parametrize("failure", ["wrong_session", "wrong_content", "assistant", "archived"])
def test_scoped_backfill_fails_closed_without_latest_row_fallback(compacted, failure):
    db, rows = compacted
    session = "session"
    content = "Create report"
    row_id = rows[0]["_row_id"]
    if failure == "wrong_session":
        db.create_session("other", source="pantheon")
        session = "other"
    elif failure == "wrong_content":
        content = "Todo snapshot"
    elif failure == "assistant":
        row_id = rows[1]["_row_id"]
        content = "Working"
    else:
        db.archive_and_compact("session", [{"role": "user", "content": "Create report"}])
    assert db.set_latest_user_api_content(
        session, content, "MUST-NOT-BE-STORED", row_id=row_id
    ) == 0
    assert all(not row.get("api_content") for row in db.get_messages("session", include_inactive=True))


@pytest.mark.parametrize("row_id", [True, False, 0, -1, "1", 1.5])
def test_scoped_backfill_rejects_invalid_row_identity(compacted, row_id):
    db, _ = compacted
    with pytest.raises(ValueError, match="row_id"):
        db.set_latest_user_api_content("session", "Create report", "WIRE", row_id=row_id)


def test_scoped_backfill_does_not_update_same_text_in_another_row(compacted):
    db, rows = compacted
    db.append_message("session", "user", content="Create report")
    assert db.set_latest_user_api_content(
        "session", "Create report", "EXACT-ROW-WIRE", row_id=rows[0]["_row_id"]
    ) == 1
    replay = db.get_messages_as_conversation("session")
    assert replay[0]["api_content"] == "EXACT-ROW-WIRE"
    assert "api_content" not in replay[-1]
