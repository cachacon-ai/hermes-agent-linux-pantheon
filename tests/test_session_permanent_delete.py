"""Permanent session deletion keeps only content-free tombstones.

These tests use real SessionDB files so they exercise SQLite lineage,
profile isolation, FTS-backed message storage, and transcript cleanup together.
"""

from __future__ import annotations

import sqlite3
import time
import json

import pytest

from hermes_state import SessionDB


def _end_as_compressed(db: SessionDB, session_id: str) -> None:
    db._execute_write(
        lambda conn: conn.execute(
            "UPDATE sessions SET ended_at = ?, end_reason = 'compression' WHERE id = ?",
            (time.time(), session_id),
        )
    )


def test_permanent_delete_erases_full_compression_chain_and_delegates_but_keeps_branch(
    tmp_path,
):
    db = SessionDB(db_path=tmp_path / "state.db")
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()

    db.create_session("root", source="api")
    db.append_message("root", "user", "secret root text")
    _end_as_compressed(db, "root")

    db.create_session("segment-1", source="api", parent_session_id="root")
    db.append_message("segment-1", "assistant", "secret middle text")
    _end_as_compressed(db, "segment-1")
    db.create_session("segment-2", source="api", parent_session_id="segment-1")
    db.append_message("segment-2", "assistant", "secret tip text")

    # Explicit branch children remain independent, along with their own
    # compression continuation, even when their parent is deleted.
    db.create_session(
        "branch",
        source="api",
        parent_session_id="segment-1",
        model_config={"_branched_from": "segment-1"},
    )
    db.append_message("branch", "user", "preserve this branch")
    _end_as_compressed(db, "branch")
    db.create_session("branch-tip", source="api", parent_session_id="branch")
    db.append_message("branch-tip", "assistant", "preserve branch continuation")

    # Delegate children are hidden work belonging to the selected logical
    # conversation, including a compression continuation of the delegate.
    db.create_session(
        "delegate",
        source="api",
        parent_session_id="root",
        model_config={"_delegate_from": "root"},
    )
    db.append_message("delegate", "assistant", "secret delegated text")
    _end_as_compressed(db, "delegate")
    db.create_session(
        "delegate-tip",
        source="api",
        parent_session_id="delegate",
        model_config={"_delegate_from": "root"},
    )
    db.append_message("delegate-tip", "assistant", "secret delegate continuation")

    expected = {"root", "segment-1", "segment-2", "delegate", "delegate-tip"}
    for session_id in expected:
        (sessions_dir / f"{session_id}.json").write_text("secret transcript")
        (sessions_dir / f"{session_id}.jsonl").write_text("secret transcript\n")
        (sessions_dir / f"request_dump_{session_id}_one.json").write_text("secret request")

    deleted = db.delete_session_permanently(
        "segment-1", sessions_dir=sessions_dir
    )
    assert set(deleted) == expected
    for session_id in expected:
        assert db.get_session(session_id) is None
        assert not (sessions_dir / f"{session_id}.json").exists()
        assert not (sessions_dir / f"{session_id}.jsonl").exists()
        assert not list(sessions_dir.glob(f"request_dump_{session_id}_*.json"))

    branch = db.get_session("branch")
    branch_tip = db.get_session("branch-tip")
    assert branch is not None and branch["parent_session_id"] is None
    assert branch_tip is not None and branch_tip["parent_session_id"] == "branch"
    with db._read_ctx() as conn:
        assert conn.execute(
            "SELECT content FROM messages WHERE session_id = 'branch'"
        ).fetchone()[0] == "preserve this branch"
        markers = conn.execute(
            "SELECT session_id, requested_session_id, deleted_at "
            "FROM permanent_session_deletions"
        ).fetchall()
        assert {row["session_id"] for row in markers} == expected
        assert {row["requested_session_id"] for row in markers} == {"segment-1"}

    # Late runtime writes cannot recreate the session or its messages.
    with pytest.raises(sqlite3.IntegrityError, match="permanently deleted"):
        db.create_session("segment-1", source="api")
    with pytest.raises(sqlite3.IntegrityError, match="permanently deleted"):
        db._execute_write(
            lambda conn: conn.execute(
                "INSERT INTO messages(session_id, role, content, timestamp) "
                "VALUES ('segment-1', 'user', 'resurrected', ?)",
                (time.time(),),
            )
        )

    # A retry finds the content-free target list and finishes any filesystem
    # cleanup left behind by an interrupted first attempt.
    retry_file = sessions_dir / "request_dump_root_retry.json"
    retry_file.write_text("secret retry dump")
    assert set(db.delete_session_permanently("segment-1", sessions_dir)) == expected
    assert not retry_file.exists()
    db.close()


def test_permanent_delete_can_fence_a_lazy_session_before_its_first_write(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    assert db.delete_session_permanently("lazy-session", allow_missing=True) == [
        "lazy-session"
    ]
    with pytest.raises(sqlite3.IntegrityError, match="permanently deleted"):
        db.create_session("lazy-session", source="api")
    db.close()


def test_permanent_delete_is_profile_scoped_even_when_session_ids_collide(tmp_path):
    default_db = SessionDB(db_path=tmp_path / "default" / "state.db")
    profile_db = SessionDB(db_path=tmp_path / "profiles" / "writer" / "state.db")
    default_db.create_session("same-id", source="api")
    default_db.append_message("same-id", "user", "keep default copy")
    profile_db.create_session("same-id", source="api")
    profile_db.append_message("same-id", "user", "delete profile copy")

    profile_db.delete_session_permanently("same-id")
    assert default_db.get_session("same-id") is not None
    assert profile_db.get_session("same-id") is None
    with default_db._read_ctx() as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM permanent_session_deletions"
        ).fetchone()[0] == 0
    default_db.close()
    profile_db.close()


def test_permanent_delete_clears_routing_target_without_reusing_generation(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("routed-session", source="api")
    db._execute_write(
        lambda conn: (
            conn.execute(
                "INSERT INTO gateway_routing(scope, session_key, entry_json, updated_at) "
                "VALUES ('api', 'peer', ?, ?)",
                (json.dumps({"session_id": "routed-session"}), time.time()),
            ),
            conn.execute(
                "INSERT INTO conversation_generations(source, session_key, generation) "
                "VALUES ('api', 'peer', 4)"
            ),
        )
    )

    db.delete_session_permanently("routed-session")

    with db._read_ctx() as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM gateway_routing WHERE session_key = 'peer'"
        ).fetchone()[0] == 0
        assert conn.execute(
            "SELECT generation FROM conversation_generations "
            "WHERE source = 'api' AND session_key = 'peer'"
        ).fetchone()[0] == 4
    db.close()
