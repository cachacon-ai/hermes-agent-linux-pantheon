"""Live reply identity is the committed row, with stable compaction lineage."""

from types import SimpleNamespace

import pytest

from agent.context_compressor import ContextCompressor, stamp_db_persisted_markers
from agent.message_metadata import REPLY_SOURCE_ROW_ID_KEY
from agent.turn_finalizer import finalize_turn
from hermes_state import SessionDB
from run_agent import AIAgent
from tests.agent.test_turn_finalizer_final_response_persistence import FakeAgent


@pytest.fixture
def durable_agent(monkeypatch, tmp_path):
    monkeypatch.setattr("hermes_cli.lifecycle.invoke_hook", lambda *_a, **_kw: [])
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("sess-test", source="cli")
    writer = AIAgent.__new__(AIAgent)
    writer._session_db = db
    writer._session_db_created = True
    writer._last_flushed_db_idx = 0
    writer.session_id = "sess-test"
    agent = FakeAgent()
    agent._session_db = db
    agent._persist_session = lambda messages, history: writer._flush_messages_to_session_db(messages, history)
    yield agent, db
    db.close()


def _finalize(agent, messages, *, answer="Same answer.", failed=False, interrupted=False):
    return finalize_turn(
        agent, final_response=answer, api_call_count=1,
        interrupted=interrupted, failed=failed, messages=messages,
        conversation_history=[], effective_task_id="task", turn_id="turn",
        user_message="question", original_user_message="question",
        _should_review_memory=False, _turn_exit_reason="text_response(finish_reason=stop)",
    )


def _turn():
    return [
        {"role": "user", "content": "question", "timestamp": 1780000000.0},
        {"role": "assistant", "content": "Same answer.", "timestamp": 1780000001.0},
    ]


def test_finalizer_identity_matches_real_flush_and_reloaded_history(durable_agent):
    agent, db = durable_agent
    messages = _turn()
    result = _finalize(agent, messages)
    reloaded = db.get_messages_as_conversation("sess-test", include_row_ids=True)

    assert result["reply_row_id"] == reloaded[-1]["_row_id"] == messages[-1]["_row_id"]
    assert messages[-1]["display_metadata"][REPLY_SOURCE_ROW_ID_KEY] == result["reply_row_id"]
    assert reloaded[-1]["display_metadata"] == messages[-1]["display_metadata"]
    assert reloaded[-1]["timestamp"] == messages[-1]["timestamp"]


def test_identical_text_and_timestamp_are_distinct_authored_replies(durable_agent):
    agent, db = durable_agent
    first = _finalize(agent, _turn())
    second = _finalize(agent, _turn())
    replies = [m for m in db.get_messages_as_conversation("sess-test", include_row_ids=True)
               if m["role"] == "assistant"]

    assert replies[0]["content"] == replies[1]["content"]
    assert replies[0]["timestamp"] == replies[1]["timestamp"]
    assert first["reply_row_id"] != second["reply_row_id"]
    assert replies[0]["display_metadata"][REPLY_SOURCE_ROW_ID_KEY] != replies[1]["display_metadata"][REPLY_SOURCE_ROW_ID_KEY]


def test_micro_compaction_changes_physical_id_but_keeps_source(durable_agent):
    agent, db = durable_agent
    messages = _turn()
    agent._persist_session(messages, [])
    original = messages[-1]["_row_id"]

    def compact(rows):
        copies = [dict(row) for row in rows]
        db.archive_and_compact("sess-test", copies, tail_count=len(copies))
        stamp_db_persisted_markers(copies)
        return copies

    agent.context_compressor = SimpleNamespace(
        last_prompt_tokens=0, _micro_compact_enabled=True, _micro_compact=compact,
    )
    result = _finalize(agent, messages)
    reloaded = db.get_messages_as_conversation("sess-test", include_row_ids=True)

    assert result["reply_row_id"] == reloaded[-1]["_row_id"]
    assert result["reply_row_id"] != original
    assert reloaded[-1]["display_metadata"][REPLY_SOURCE_ROW_ID_KEY] == original
    assert messages[-1]["display_metadata"][REPLY_SOURCE_ROW_ID_KEY] == original


def test_hooks_and_verifier_footer_do_not_change_persisted_reply_identity(durable_agent, monkeypatch):
    agent, db = durable_agent
    agent._turn_failed_file_mutations = {"file": "failed"}
    agent._file_mutation_verifier_enabled = lambda: True
    agent._format_file_mutation_failure_footer = lambda failures: "Verifier footer."
    monkeypatch.setattr("hermes_cli.lifecycle.invoke_hook", lambda name, **kw:
                        [kw["response_text"] + " Hook footer."] if name == "transform_llm_output" else [])

    result = _finalize(agent, _turn())
    reloaded = db.get_messages_as_conversation("sess-test", include_row_ids=True)

    assert result["final_response"] == "Same answer.\n\nVerifier footer. Hook footer."
    assert reloaded[-1]["content"] == "Same answer."
    assert result["reply_row_id"] == reloaded[-1]["_row_id"]


@pytest.mark.parametrize("case", ["failed", "interrupted", "unsaved", "empty", "disabled", "persistence_error"])
def test_missing_real_committed_reply_never_gets_an_identity(durable_agent, case):
    agent, _db = durable_agent
    messages = _turn()
    if case == "unsaved":
        agent._persist_session = lambda *_: None
    if case == "persistence_error":
        def reject(*_):
            raise OSError("disk full")
        agent._persist_session = reject
    if case == "disabled":
        agent._persist_disabled = True
    if case == "empty":
        messages[-1]["content"] = ""
    result = _finalize(agent, messages, answer="" if case == "empty" else "Same answer.",
                       failed=case == "failed", interrupted=case == "interrupted")
    assert "reply_row_id" not in result


def test_legacy_row_decoding_carries_exact_source_through_future_compactions(durable_agent):
    _agent, db = durable_agent
    original = db.append_message("sess-test", "assistant", "Legacy answer", timestamp=1780000001.0)
    db._execute_write(lambda conn: conn.execute("UPDATE messages SET display_metadata = NULL WHERE id = ?", (original,)))
    rows = db.get_messages_as_conversation("sess-test", include_row_ids=True)
    assert rows[-1]["display_metadata"][REPLY_SOURCE_ROW_ID_KEY] == original

    for _ in range(2):
        db.archive_and_compact("sess-test", rows, tail_count=len(rows))
        rows = db.get_messages_as_conversation("sess-test", include_row_ids=True)
        assert rows[-1]["_row_id"] != original
        assert rows[-1]["display_metadata"][REPLY_SOURCE_ROW_ID_KEY] == original


def test_legacy_concurrent_watermark_tail_keeps_its_original_source(durable_agent):
    _agent, db = durable_agent
    db.append_message("sess-test", "user", "old question", timestamp=1780000000.0)
    watermark = db.get_active_message_watermark("sess-test")
    original = db.append_message("sess-test", "assistant", "Concurrent answer", timestamp=1780000001.0)
    db._execute_write(lambda conn: conn.execute("UPDATE messages SET display_metadata = NULL WHERE id = ?", (original,)))

    db.archive_and_compact("sess-test", [{"role": "user", "content": "summary"}], watermark=watermark)
    rows = db.get_messages_as_conversation("sess-test", include_row_ids=True)

    assert rows[-1]["content"] == "Concurrent answer"
    assert rows[-1]["_row_id"] != original
    assert rows[-1]["display_metadata"][REPLY_SOURCE_ROW_ID_KEY] == original


def test_rotation_watermark_clone_preserves_legacy_source(durable_agent):
    _agent, db = durable_agent
    db.append_message("sess-test", "user", "old question", timestamp=1780000000.0)
    watermark = db.get_active_message_watermark("sess-test")
    original = db.append_message("sess-test", "assistant", "Concurrent answer", timestamp=1780000001.0)
    db._execute_write(lambda conn: conn.execute("UPDATE messages SET display_metadata = NULL WHERE id = ?", (original,)))

    db.publish_compression_child(
        parent_session_id="sess-test", child_session_id="child", source="cli",
        messages=[{"role": "user", "content": "summary"}], watermark=watermark,
        require_compression_lease=False,
    )
    rows = db.get_messages_as_conversation("child", include_row_ids=True)

    assert rows[-1]["_row_id"] != original
    assert rows[-1]["display_metadata"][REPLY_SOURCE_ROW_ID_KEY] == original


def test_reply_source_preserves_other_display_metadata(durable_agent):
    agent, db = durable_agent
    messages = _turn()
    messages[-1]["display_metadata"] = {"reactions": {"thumbsup": ["user"]}}

    _finalize(agent, messages)
    row = db.get_messages_as_conversation("sess-test", include_row_ids=True)[-1]

    assert row["display_metadata"]["reactions"] == {"thumbsup": ["user"]}
    assert row["display_metadata"][REPLY_SOURCE_ROW_ID_KEY] == row["_row_id"]


def test_blank_tail_repair_finds_exact_clone_among_equal_timestamp_rows(durable_agent):
    agent, db = durable_agent
    first = db.append_message("sess-test", "assistant", "", timestamp=1780000001.0)
    second = db.append_message("sess-test", "assistant", "", timestamp=1780000001.0)
    before = db.get_messages_as_conversation("sess-test", include_row_ids=True)
    stale_first = dict(before[0])
    copies = [dict(row) for row in before]
    db.archive_and_compact("sess-test", copies, tail_count=len(copies))
    stale_first["content"] = "Recovered first answer"
    stale_first.pop("_db_persisted", None)

    agent._persist_session([stale_first], [])
    after = db.get_messages_as_conversation("sess-test", include_row_ids=True)
    by_source = {row["display_metadata"][REPLY_SOURCE_ROW_ID_KEY]: row for row in after}

    assert len(after) == 2
    assert by_source[first]["content"] == "Recovered first answer"
    assert by_source[second]["content"] == ""
    assert stale_first["_row_id"] == by_source[first]["_row_id"]
    assert stale_first["display_metadata"][REPLY_SOURCE_ROW_ID_KEY] == first


def test_rolled_back_batch_never_marks_live_answer_committed(durable_agent, monkeypatch):
    agent, db = durable_agent
    insert = db._insert_message_rows

    def reject_after_insert(*args, **kwargs):
        insert(*args, **kwargs)
        raise OSError("write interrupted before commit")

    monkeypatch.setattr(db, "_insert_message_rows", reject_after_insert)
    messages = _turn()

    result = _finalize(agent, messages)

    assert "reply_row_id" not in result
    assert not messages[-1].get("_db_persisted")
    assert db.get_messages_as_conversation("sess-test", include_row_ids=True) == []


def test_rolled_back_micro_compaction_cannot_advertise_uncommitted_row(durable_agent, monkeypatch):
    agent, db = durable_agent
    messages = _turn()
    agent._persist_session(messages, [])
    original = messages[-1]["_row_id"]
    insert = db._insert_message_rows

    def reject_after_insert(*args, **kwargs):
        insert(*args, **kwargs)
        raise OSError("compaction interrupted before commit")

    monkeypatch.setattr(db, "_insert_message_rows", reject_after_insert)
    compressor = SimpleNamespace(_session_db=db, _session_id="sess-test")

    def compact(rows):
        # Real micro-compaction carries the original suffix dictionaries.
        ContextCompressor._sync_micro_compact_to_db(compressor, rows)
        return rows

    agent.context_compressor = SimpleNamespace(
        last_prompt_tokens=0, _micro_compact_enabled=True, _micro_compact=compact,
    )
    result = _finalize(agent, messages)

    assert messages[-1]["_row_id"] != original
    assert messages[-1]["_db_persisted"] is True
    assert db.get_messages_as_conversation("sess-test", include_row_ids=True)[-1]["_row_id"] == original
    assert "reply_row_id" not in result


@pytest.mark.parametrize("change", [
    {"active": 0}, {"content": "Other reply"},
    {"timestamp": 1780000002.0}, {"display_kind": "hidden"},
    {"display_metadata": '{"_reply_source_row_id": 999}'},
])
def test_stale_live_marker_cannot_advertise_different_database_row(durable_agent, change):
    agent, db = durable_agent
    messages = _turn()
    agent._persist_session(messages, [])
    field, value = next(iter(change.items()))
    db._execute_write(lambda conn: conn.execute(
        f"UPDATE messages SET {field} = ? WHERE id = ?", (value, messages[-1]["_row_id"]),
    ))

    result = _finalize(agent, messages)

    assert "reply_row_id" not in result


def test_identity_read_failure_keeps_successful_turn_successful(durable_agent, monkeypatch):
    agent, db = durable_agent

    def unavailable(*_args, **_kwargs):
        raise OSError("identity read unavailable")

    monkeypatch.setattr(db, "get_messages_around", unavailable)
    result = _finalize(agent, _turn())

    assert result["final_response"] == "Same answer."
    assert not result.get("failed")
    assert "cleanup_errors" not in result
    assert "reply_row_id" not in result
    assert len(db.get_messages_as_conversation("sess-test")) == 2
