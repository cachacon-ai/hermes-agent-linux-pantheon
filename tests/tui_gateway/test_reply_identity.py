"""Persisted final reply metadata reaches the actual message.complete frame."""

import pytest

from agent.message_metadata import REPLY_SOURCE_ROW_ID_KEY
from tests.agent.test_reply_identity import durable_agent, _finalize, _turn
from tests.tui_gateway.test_prompt_submit_clock_context import turn_env, _session
from tui_gateway import server


def test_real_prompt_submit_emits_exact_identity_and_canonical_reply_text(
    durable_agent, turn_env, monkeypatch,
):
    agent, db = durable_agent
    monkeypatch.setattr("hermes_cli.lifecycle.invoke_hook", lambda name, **kw:
                        [kw["response_text"] + " Delivery hook."] if name == "transform_llm_output" else [])
    result = _finalize(agent, _turn())
    agent.run_conversation = lambda *_a, **_kw: result
    frames = []
    monkeypatch.setattr(server, "_emit", lambda event, sid, payload=None:
                        frames.append(server._event_frame(event, sid, payload)))

    server._run_prompt_submit("request", "runtime-session", _session(agent), "question")

    completions = [frame["params"]["payload"] for frame in frames
                   if frame["params"]["type"] == "message.complete"]
    assert len(completions) == 1
    completion = completions[0]
    history = server._history_to_messages(
        db.get_messages_as_conversation("sess-test", include_row_ids=True),
    )[-1]
    assert completion["reply_identity_version"] == 1
    assert completion["row_id"] == history["row_id"]
    assert completion["source_row_id"] == history["source_row_id"]
    assert completion["timestamp"] == history["timestamp"]
    assert completion["reply_text"] == history["text"] == "Same answer."
    assert completion["text"] == "Same answer. Delivery hook."


def test_gateway_projects_compacted_reply_with_stable_source(durable_agent):
    agent, db = durable_agent
    messages = _turn()
    original = _finalize(agent, messages)["reply_row_id"]
    copies = [dict(message) for message in messages]
    db.archive_and_compact("sess-test", copies, tail_count=len(copies))
    current = db.get_messages_as_conversation("sess-test", include_row_ids=True)
    result = {"reply_row_id": current[-1]["_row_id"], "messages": current}

    identity = server._persisted_reply_identity(result)
    projected = server._history_to_messages(current)[-1]

    assert identity["row_id"] == projected["row_id"] != original
    assert identity["source_row_id"] == projected["source_row_id"] == original
    assert identity["reply_text"] == projected["text"]


@pytest.mark.parametrize("payload", [None, {"text": "Error", "status": "error"}, {"text": "Synthetic"}])
def test_all_complete_frames_advertise_capability_without_inventing_a_row(payload):
    original = dict(payload) if payload else None
    frame = server._event_frame("message.complete", "session", payload)

    assert frame["params"]["payload"]["reply_identity_version"] == 1
    assert "row_id" not in frame["params"]["payload"]
    assert payload == original
    assert "reply_identity_version" not in server._event_frame("message.delta", "session", {})["params"]["payload"]


@pytest.mark.parametrize("row_id", [0, -1, True, "1", 2**53])
def test_invalid_or_unsafe_wire_identity_is_omitted(row_id):
    result = {"reply_row_id": row_id, "messages": [{
        "role": "assistant", "content": "reply", "_row_id": row_id,
        "_db_persisted": True, "timestamp": 1780000001.0,
    }]}
    assert server._persisted_reply_identity(result) == {}


@pytest.mark.parametrize("flag", ["failed", "error", "interrupted"])
def test_failed_result_cannot_advertise_a_real_reply(flag):
    result = _result()
    result[flag] = True
    assert server._persisted_reply_identity(result) == {}


@pytest.mark.parametrize("change", [
    {"_db_persisted": False}, {"content": ""}, {"display_kind": "hidden"},
    {"timestamp": float("nan")}, {"timestamp": float("inf")},
])
def test_unsaved_hidden_empty_or_invalid_timestamp_reply_is_omitted(change):
    result = _result()
    result["messages"][0].update(change)
    assert server._persisted_reply_identity(result) == {}


def test_exact_row_selection_ignores_identical_assistant_text():
    result = _result()
    previous = dict(result["messages"][0], _row_id=1,
                    display_metadata={REPLY_SOURCE_ROW_ID_KEY: 1})
    result["messages"].insert(0, previous)

    identity = server._persisted_reply_identity(result)

    assert identity["row_id"] == identity["source_row_id"] == 2


def test_malformed_legacy_metadata_does_not_crash_history_projection():
    result = _result()
    result["messages"][0]["display_metadata"] = "legacy invalid metadata"
    identity = server._persisted_reply_identity(result)
    assert identity["row_id"] == 2
    assert "source_row_id" not in identity


def _result():
    return {"reply_row_id": 2, "messages": [{
        "role": "assistant", "content": "reply", "_row_id": 2,
        "_db_persisted": True, "timestamp": 1780000001.0,
        "display_metadata": {REPLY_SOURCE_ROW_ID_KEY: 2},
    }]}
