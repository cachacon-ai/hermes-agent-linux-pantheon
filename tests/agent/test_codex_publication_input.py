"""Native publication guidance crosses the real prologue/adapter boundary."""
from copy import deepcopy
import json
from pathlib import Path
import re
from types import SimpleNamespace

import pytest

from agent.conversation_loop import run_conversation
from run_agent import AIAgent
from hermes_state import SessionDB


@pytest.fixture
def native_runtime(tmp_path, monkeypatch):
    monkeypatch.setattr("agent.title_generator.maybe_auto_title", lambda *a, **k: None)
    db = SessionDB(tmp_path / "profile" / "state.db")
    db.create_session("native-publication", source="pantheon", model_config={"artifact_delivery_version": 1})
    agent = AIAgent(
        api_key="test-key", base_url="https://openrouter.ai/api/v1",
        quiet_mode=True, skip_context_files=True, skip_memory=True,
        session_db=db, session_id="native-publication",
    )
    agent._session_db_created = True
    agent.api_mode = "codex_app_server"
    agent.compression_enabled = False
    agent._pantheon_artifact_delivery_version = 1
    agent._pantheon_profile_home = str(tmp_path / "profile")
    agent._pantheon_profile = "test"
    agent.session_cwd = str(tmp_path)
    agent._memory_nudge_interval = 0
    agent._skill_nudge_interval = 0
    agent._prompt_clock_context = "[Current time: 2026-10-20 12:00:00 UTC]"
    sessions = []

    class Adapter:
        supported = True
        publish = False

        def __init__(self, **kwargs):
            self.options = kwargs
            self.inputs = []
            self.artifact_delivery_available = self.supported
            sessions.append(self)

        def run_turn(self, *, user_input):
            self.inputs.append(deepcopy(user_input))
            if self.publish:
                outside = tmp_path / "outside.html"
                outside.write_text("<!doctype html><html>outside</html>")
                rejected = self.options["dynamic_tool_handler"]({
                    "callId": "outside-call", "arguments": {"path": str(outside)},
                })
                assert rejected["success"] is False
                error = json.loads(rejected["contentItems"][0]["text"])
                assert error["error"] == "source_outside_root"
                root = re.search(r"write completed deliverables for this turn in (.+?)\. Call", _text(user_input)).group(1)
                source = Path(root) / "report.html"
                source.write_text("<!doctype html><html>report</html>")
                captured = self.options["dynamic_tool_handler"]({
                    "callId": "publication-call", "arguments": {"path": str(source)},
                })
                assert captured["success"] is True, captured
            return SimpleNamespace(
                interrupted=False, error=None, thread_id="native-thread",
                turn_id=f"native-turn-{len(self.inputs)}", tool_iterations=0,
                final_text="Done", should_retire=False,
                projected_messages=[{"role": "assistant", "content": "Done"}],
            )

        def close(self):
            pass

    monkeypatch.setattr(
        "agent.transports.codex_app_server_session.CodexAppServerSession", Adapter,
    )
    monkeypatch.setattr("agent.auxiliary_client.set_runtime_main", lambda *a, **k: None)
    monkeypatch.setattr(
        "hermes_cli.lifecycle.invoke_hook",
        lambda name, **kwargs: [{"context": "PRIVATE_PROVIDER_PLUGIN_CONTEXT"}]
        if name == "pre_llm_call" else [],
    )
    yield agent, Adapter, sessions
    db.close()


def _text(content):
    if isinstance(content, str):
        return content
    return "\n".join(part.get("text", "") for part in content if isinstance(part, dict))


@pytest.mark.parametrize("content", [
    "make a report",
    [{"type": "text", "text": "make a report"},
     {"type": "image_url", "image_url": {"url": "data:image/png;base64,eA=="}}],
])
def test_native_input_gets_current_publication_root_without_dirty_history(native_runtime, content):
    agent, _, sessions = native_runtime
    original = deepcopy(content)
    result = run_conversation(agent, content)
    assert result["completed"] is True
    wire = sessions[0].inputs[0]
    text = _text(wire)
    assert agent._pantheon_turn_output_root in text
    assert text.count("[Pantheon file delivery:") == 1
    assert text.count(agent._turn_prompt_clock) == 1
    assert "pantheon_publish_artifact" in text
    assert "PRIVATE_PROVIDER_PLUGIN_CONTEXT" not in text
    assert content == original
    user = next(message for message in result["messages"] if message["role"] == "user")
    assert user["content"] == original
    assert "[Pantheon file delivery:" not in agent._cached_system_prompt


def test_native_reused_session_gets_new_root_only(native_runtime):
    agent, _, sessions = native_runtime
    first = run_conversation(agent, "first")
    previous_root = agent._pantheon_turn_output_root
    second = run_conversation(agent, "second", conversation_history=first["messages"])
    assert second["completed"] is True
    assert len(sessions) == 1
    assert agent._pantheon_turn_output_root != previous_root
    assert previous_root not in _text(sessions[0].inputs[1])
    assert agent._pantheon_turn_output_root in _text(sessions[0].inputs[1])


def test_native_guided_capture_commits_bytes_and_rejects_external_root(native_runtime):
    agent, adapter, _ = native_runtime
    adapter.publish = True
    result = run_conversation(agent, "publish report")
    assert result["completed"] is True
    artifact = result["artifacts"][0]
    assert agent._session_db.artifact_content(agent.session_id, artifact["artifact_id"])[1] == b"<!doctype html><html>report</html>"
    rows = agent._session_db.get_messages_as_conversation(agent.session_id)
    user = next(row for row in rows if row["role"] == "user")
    assert user["content"] == "publish report"


@pytest.mark.parametrize("disabled", [False, True])
def test_native_unavailable_first_turn_has_no_publication_guidance(native_runtime, disabled):
    agent, adapter, sessions = native_runtime
    if disabled:
        agent._pantheon_artifact_delivery_version = 0
    else:
        adapter.supported = False
    result = run_conversation(agent, "make a report")
    assert result["completed"] is True
    assert "[Pantheon file delivery:" not in _text(sessions[0].inputs[0])
    assert agent._pantheon_artifact_delivery_version == 0
    if not disabled:
        assert agent._pantheon_artifact_unavailable_code == "unsupported_protocol"
