"""Publication context survives real compressor summary assembly (PAN-68)."""
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.context_compressor import ContextCompressor, COMPRESSED_SUMMARY_METADATA_KEY
from tests.agent.test_api_content_sidecar import _FakeAgent, _build, wire_env
from hermes_state import SessionDB


def _artifact_agent(tmp_path):
    agent = _FakeAgent()
    agent.session_id = tmp_path.name
    agent._pantheon_artifact_delivery_version = 1
    agent._pantheon_profile_home = str(tmp_path)
    return agent


def _merge_compressor():
    with patch("agent.context_compressor.get_model_context_length", return_value=100000):
        compressor = ContextCompressor(model="test/model", protect_first_n=0,
                                       protect_last_n=0, quiet_mode=True)
        _ = compressor.context_length
    # Force the summary-to-current-user merge selection and stub model output.
    # The compressor's copy/merge/drop logic is real.
    compressor._find_tail_cut_by_tokens = lambda messages, head_end: len(messages) - 1
    compressor._generate_summary = lambda *a, **kw: compressor._with_summary_prefix(
        "Previously investigated report sources."
    )
    return compressor


@pytest.fixture
def session_db(tmp_path):
    with SessionDB(tmp_path / "state.db") as db:
        yield db


@pytest.mark.parametrize("trigger", ["preflight", "idle"])
@pytest.mark.parametrize("in_place", [False, True])
@pytest.mark.parametrize("with_context", [False, True])
def test_prologue_restores_publication_after_summary_merge(tmp_path, session_db, with_context, in_place, trigger):
    agent = _artifact_agent(tmp_path)
    agent.compression_enabled = True
    if in_place:
        session_db.create_session(agent.session_id, source="pantheon")
        agent._session_db = session_db
        agent._session_db_created = True
    if with_context:
        agent._memory_manager = SimpleNamespace(
            on_turn_start=lambda *a: None,
            prefetch_all=lambda query: "RECALLED-MEMORY",
            describe_recall=lambda: "",
        )
    compressor = _merge_compressor()
    if trigger == "idle":
        agent.compression_idle_compact_after_seconds = 1
        agent._last_activity_ts = 0
        compressor.summary_target_ratio = 0.001
    decisions = iter([trigger == "preflight", False, False])
    compressor.should_compress = lambda tokens: next(decisions, False)
    compressor.should_defer_preflight_to_real_usage = lambda tokens: False
    agent.context_compressor = compressor
    observed = {}

    def compress(messages, system, **kwargs):
        observed["before"] = dict(messages[-1])
        compressed = compressor.compress(messages, force=True)
        observed["after"] = dict(compressed[-1])
        if in_place:
            session_db.archive_and_compact(agent.session_id, compressed, tail_count=len(compressed))
            agent._last_compaction_in_place = True
        return compressed, "SYSTEM"

    agent._compress_context = compress
    # A user-leading tail with no assistant bridge exercises the alternation-safe
    # merge path used by image-only/repaired transcripts too.
    history = [{"role": "user" if i % 2 == 0 else "assistant",
                "content": "Earlier work " * 100} for i in range(12)]
    history.append({"role": "user", "content": "Earlier related request"})
    with patch("hermes_cli.plugins.invoke_hook", return_value=(
        [{"context": "PLUGIN-CTX"}] if with_context else []
    )):
        ctx = _build(agent, user_message="Generate the report from these sources", conversation_history=history)
    current = ctx.messages[ctx.current_turn_user_idx]
    assert observed["after"].get(COMPRESSED_SUMMARY_METADATA_KEY)
    assert "api_content" not in observed["after"]  # stale bytes never bypass summary
    assert "Previously investigated report sources." in current["content"]
    assert "[Pantheon file delivery:" not in current["content"]
    assert current["api_content"].count("[Pantheon file delivery:") == 1
    assert agent._pantheon_turn_output_root in current["api_content"]
    if with_context:
        assert current["api_content"].count("PLUGIN-CTX") == 1
        assert current["api_content"].count("RECALLED-MEMORY") == 1
    assert current["api_content"] == agent.api_content_at_persist
    if in_place:
        replay = session_db.get_messages_as_conversation(agent.session_id)[-1]
        assert replay["content"] == current["content"]
        assert replay["api_content"] == current["api_content"]


@pytest.mark.parametrize("multimodal", [False, True])
def test_publication_is_staged_once_per_turn_without_early_sidecar(tmp_path, multimodal):
    agent = _artifact_agent(tmp_path)
    content = [{"type": "text", "text": "hello"},
               {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}] if multimodal else "hello"
    from agent.pantheon_artifacts import prepare_turn_output
    with patch("agent.pantheon_artifacts.prepare_turn_output", wraps=prepare_turn_output) as prepare:
        with patch("hermes_cli.plugins.invoke_hook", return_value=[]):
            ctx = _build(agent, user_message=content, summarize_user_message_for_log=lambda value: "hello")
        prepare.assert_called_once_with(agent)
    old_note = agent._turn_pantheon_file_delivery
    assert old_note and agent._pantheon_turn_output_root in old_note
    assert ctx.messages[ctx.current_turn_user_idx]["content"] == content
    if multimodal:
        assert "api_content" not in ctx.messages[ctx.current_turn_user_idx]
    else:
        assert ctx.messages[ctx.current_turn_user_idx]["api_content"] == "hello\n\n" + old_note
    agent._pantheon_artifact_delivery_version = 0
    with patch("hermes_cli.plugins.invoke_hook", return_value=[]):
        next_ctx = _build(agent)
    assert agent._turn_pantheon_file_delivery == ""
    assert "api_content" not in next_ctx.messages[next_ctx.current_turn_user_idx]


@pytest.mark.parametrize("trigger", ["preflight", "idle"])
@pytest.mark.parametrize("in_place", [False, True])
def test_real_prologue_compaction_preserves_next_turn_wire_replay(
    wire_env, monkeypatch, tmp_path, trigger, in_place,
):
    from unittest.mock import MagicMock
    from tests.agent.test_api_content_sidecar import _chat_requests, _user_messages

    monkeypatch.setattr("agent.title_generator.generate_title", lambda *a, **kw: None)
    monkeypatch.setattr("agent.auxiliary_client.get_text_auxiliary_client", lambda *a, **kw: (None, None))
    monkeypatch.setattr("agent.auxiliary_client._try_configured_fallback_for_unavailable_client", lambda *a, **kw: (None, None, ""))
    make_agent, handler, db, sid = wire_env
    agent = make_agent()
    agent._pantheon_artifact_delivery_version = 1
    agent._pantheon_profile_home = str(tmp_path)
    agent.compression_enabled = True
    agent.compression_in_place = in_place
    compressor = _merge_compressor()
    compressor.threshold_tokens = 50_000
    if trigger == "idle":
        agent.compression_idle_compact_after_seconds = 1
        agent._last_activity_ts = 0
        # Preserve an idle timestamp at the prologue boundary; runtime heartbeat
        # updates are separate from the publication composition under test.
        monkeypatch.setattr(agent, "_touch_activity", lambda *a, **kw: None)
        compressor.summary_target_ratio = 0.001
    decisions = iter([trigger == "preflight"])
    compressor.should_compress = lambda tokens: next(decisions, False)
    compressor.should_defer_preflight_to_real_usage = lambda tokens: False
    agent.context_compressor = compressor
    agent._memory_manager = MagicMock()
    agent._memory_manager.prefetch_all.return_value = "MEMORY-MARKER"
    agent._memory_manager.describe_recall.return_value = ""
    agent._memory_manager.has_tool.return_value = False
    monkeypatch.setattr("agent.turn_context.estimate_request_tokens_rough", lambda *a, **kw: 90_000)
    history = [{"role": "user" if i % 2 == 0 else "assistant", "content": "Earlier work " * 100} for i in range(12)]
    history.append({"role": "user", "content": "Earlier related request"})
    agent._ensure_db_session()
    for message in history:
        db.append_message(sid, message["role"], content=message["content"])
    result = agent.run_conversation("Generate the report from these sources",
                                    conversation_history=history, task_id="prologue-compaction")
    assert result["completed"] is True
    assert compressor.compression_count == 1
    assert (agent.session_id == sid) is in_place
    sent = _user_messages(_chat_requests(handler)[0])[-1]["content"]
    for marker in (agent._pantheon_turn_output_root, "PLUGIN-CTX", "MEMORY-MARKER"):
        assert sent.count(marker) == 1
    replay = db.get_messages_as_conversation(agent.session_id)
    stored = next(m for m in reversed(replay) if m["role"] == "user")
    current = next(m for m in reversed(result["messages"]) if m["role"] == "user")
    assert stored.get("api_content") == current["api_content"], repr(stored)
    assert "[Pantheon file delivery:" not in stored["content"]
    handler.captured_requests = []
    agent.compression_enabled = False
    agent.run_conversation("A separate follow-up request", conversation_history=replay, task_id="replay")
    assert _user_messages(_chat_requests(handler)[0])[0]["content"] == sent
