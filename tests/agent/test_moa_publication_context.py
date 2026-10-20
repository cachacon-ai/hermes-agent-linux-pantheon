"""Publication, recall and plugin context at real reference/provider boundaries."""
from copy import deepcopy
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tests.agent.test_api_content_sidecar import (
    wire_env, _chat_requests, _tc_resp, _text_resp, _user_messages,
)


@pytest.mark.parametrize("with_context", [True, False])
@pytest.mark.parametrize("publication", [True, False])
@pytest.mark.parametrize("multimodal", [False, True])
def test_moa_reference_and_provider_context_each_iteration(wire_env, monkeypatch, tmp_path, publication, multimodal, with_context):
    from agent import moa_loop
    from agent.message_content import flatten_message_text

    # Keep all auxiliary model/runtime boundaries deterministic and offline.
    monkeypatch.setattr("agent.title_generator.generate_title", lambda *a, **kw: None)
    monkeypatch.setattr("agent.auxiliary_client.get_text_auxiliary_client", lambda *a, **kw: (None, None))
    monkeypatch.setattr("agent.auxiliary_client._try_configured_fallback_for_unavailable_client", lambda *a, **kw: (None, None, ""))
    make_agent, handler, db, sid = wire_env
    agent = make_agent()
    agent._pantheon_artifact_delivery_version = int(publication)
    agent._pantheon_profile_home = str(tmp_path)
    agent._memory_manager = MagicMock()
    agent._memory_manager.prefetch_all.return_value = "MEMORY-MARKER" if with_context else ""
    if not with_context:
        monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *a, **kw: [])
    agent._memory_manager.describe_recall.return_value = ""
    agent._memory_manager.has_tool.return_value = False
    calls = []

    def model_call(**kwargs):
        calls.append(deepcopy(kwargs))
        text = "REFERENCE-MARKER" if kwargs["task"] == "moa_reference" else "SYNTHESIS-MARKER"
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])

    monkeypatch.setattr(moa_loop, "call_llm", model_call)
    monkeypatch.setattr(moa_loop, "_slot_runtime", lambda slot: {"provider": "openai-compat", "model": slot["model"]})
    handler.response_queue.extend([
        _tc_resp("read_file", '{"path": "/nonexistent-path"}'), _text_resp("done"),
    ])
    request = "Create a report about the project status"
    user_input = [{"type": "text", "text": request}] if multimodal else request
    result = agent.run_conversation(user_input, conversation_history=[], task_id="moa-publication", moa_config={
        "reference_models": [{"provider": "openai-compat", "model": "advisor"}],
        "aggregator": {"provider": "openai-compat", "model": "synthesizer"},
    })
    reference_calls = [c for c in calls if c["task"] == "moa_reference"]
    main_calls = _chat_requests(handler)
    assert len(reference_calls) == len(main_calls) == 2
    for reference, main in zip(reference_calls, main_calls):
        ref_text = "\n".join(flatten_message_text(m["content"]) for m in _user_messages(reference))
        main_text = flatten_message_text(_user_messages(main)[-1]["content"])
        for text in (ref_text, main_text):
            assert text.count(request) == 1
            assert text.count("PLUGIN-CTX") == (1 if with_context and not multimodal else 0)
            # Existing recall intentionally skips multimodal queries.
            assert text.count("MEMORY-MARKER") == (1 if with_context and not multimodal else 0)
            if publication:
                assert text.count(agent._pantheon_turn_output_root) == 1
            else:
                assert "[Pantheon file delivery:" not in text
                assert not getattr(agent, "_pantheon_turn_output_root", None)
        assert "SYNTHESIS-MARKER" not in ref_text
        assert main_text.count("SYNTHESIS-MARKER") == 1
        assert all("api_content" not in m for m in main["messages"])
    current = next(m for m in result["messages"] if m["role"] == "user")
    assert current["content"] == user_input
    assert "api_content" not in current
    rows = [row for row in db.get_messages(sid) if row["role"] == "user"]
    assert not rows[0]["api_content"]


@pytest.mark.parametrize("trigger", ["pre_api", "overflow", "post_tool"])
@pytest.mark.parametrize("moa", [False, True])
@pytest.mark.parametrize("in_place", [False, True])
def test_later_pre_api_compaction_rebuilds_current_context_and_replay(
    wire_env, monkeypatch, tmp_path, moa, in_place, trigger,
):
    from agent import moa_loop
    from tests.agent.test_pantheon_compaction_guidance import _merge_compressor
    from agent.pantheon_artifacts import prepare_turn_output

    # Keep all auxiliary model/runtime boundaries deterministic and offline.
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
    # The real model-summary boundary supplies this prefix. It is also the
    # durable provenance marker used after DB reload drops internal flags.
    compressor._generate_summary = lambda *a, **kw: compressor._with_summary_prefix(
        "Previously investigated report sources."
    )
    compressor.threshold_tokens = 50_000
    decisions = iter([False, True] if trigger == "pre_api" else ([False, False, True] if trigger == "post_tool" else []))
    compressor.should_compress = lambda tokens: next(decisions, False)
    compressor.should_defer_preflight_to_real_usage = lambda tokens: False
    agent.context_compressor = compressor
    agent._memory_manager = MagicMock()
    agent._memory_manager.prefetch_all.return_value = "MEMORY-MARKER"
    agent._memory_manager.describe_recall.return_value = ""
    agent._memory_manager.has_tool.return_value = False
    monkeypatch.setattr("agent.turn_context.estimate_request_tokens_rough", lambda *a, **kw: 1000)
    monkeypatch.setattr("agent.conversation_loop.estimate_request_tokens_rough", lambda *a, **kw: 1000 if compressor.compression_count else 90_000)
    monkeypatch.setattr("agent.conversation_loop.estimate_messages_tokens_rough", lambda *a, **kw: 1000 if compressor.compression_count else 90_000)
    auxiliary_calls = []
    if trigger == "overflow":
        original_post = handler.do_POST
        overflowed = []

        def post_with_overflow(self):
            if self.path.endswith("/chat/completions") and not overflowed:
                overflowed.append(True)
                self.rfile.read(int(self.headers.get("Content-Length", 0)))
                body = json.dumps({"error": {"message": "maximum context length exceeded", "type": "invalid_request_error", "code": "context_length_exceeded"}}).encode()
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            return original_post(self)

        monkeypatch.setattr(handler, "do_POST", post_with_overflow)
    elif trigger == "post_tool":
        tool_response = _tc_resp("read_file", '{"path": "/nonexistent-path"}')
        tool_response["usage"]["prompt_tokens"] = 90_000
        handler.response_queue.extend([tool_response, _text_resp("done")])

    def model_call(**kwargs):
        auxiliary_calls.append(deepcopy(kwargs))
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="SYNTHESIS-MARKER"))])

    monkeypatch.setattr(moa_loop, "call_llm", model_call)
    monkeypatch.setattr(moa_loop, "_slot_runtime", lambda slot: {"provider": "openai-compat", "model": slot["model"]})
    history = [{"role": "user" if i % 2 == 0 else "assistant", "content": "Earlier work " * 100} for i in range(12)]
    agent._ensure_db_session()
    for message in history:
        db.append_message(sid, message["role"], content=message["content"])
    prepared_roots = []

    def prepare_once(agent):
        note = prepare_turn_output(agent)
        prepared_roots.append(agent._pantheon_turn_output_root)
        return note

    monkeypatch.setattr("agent.pantheon_artifacts.prepare_turn_output", prepare_once)
    result = agent.run_conversation(
        "Generate the report from these sources", conversation_history=history,
        task_id="later-compaction", moa_config=({
            "reference_models": [{"provider": "openai-compat", "model": "advisor"}],
            "aggregator": {"provider": "openai-compat", "model": "synthesizer"},
        } if moa else None),
    )
    assert result["completed"] is True
    assert compressor.compression_count == 1
    assert len(prepared_roots) == 1
    calls = _chat_requests(handler)
    assert len(calls) == (2 if trigger == "post_tool" else 1)
    sent = _user_messages(calls[-1])[-1]["content"]
    assert any("Previously investigated report sources." in str(m.get("content")) for m in calls[-1]["messages"])
    assert "Generate the report from these sources" in sent
    for marker in (prepared_roots[0], "PLUGIN-CTX", "MEMORY-MARKER"):
        assert sent.count(marker) == 1
    current = next(m for m in reversed(result["messages"]) if m["role"] == "user")
    assert "[Pantheon file delivery:" not in current["content"]
    if moa:
        references = [c for c in auxiliary_calls if c["task"] == "moa_reference"]
        assert len(references) == 2  # first/speculative pass plus post-compaction rebuild
        reference = next(m["content"] for m in references[-1]["messages"] if prepared_roots[0] in str(m.get("content")))
        for marker in (prepared_roots[0], "PLUGIN-CTX", "MEMORY-MARKER"):
            assert reference.count(marker) == 1
        assert "SYNTHESIS-MARKER" not in reference
        assert "api_content" not in current
    else:
        assert current["api_content"] in sent
        replay = db.get_messages_as_conversation(agent.session_id)
        stored = next(m for m in reversed(replay) if m["role"] == "user")
        assert "[Pantheon file delivery:" not in stored["content"]
        assert stored.get("api_content") == current["api_content"], repr(stored)
        # Compare the actual provider replay after canonicalization (summary
        # and adjacent user rows can be combined on the transport).
        handler.captured_requests = []
        agent.compression_enabled = False
        agent.run_conversation("A separate follow-up request", conversation_history=replay, task_id="replay")
        replayed = next(m["content"] for m in _user_messages(_chat_requests(handler)[0]) if prepared_roots[0] in str(m["content"]))
        assert replayed == sent
