"""Tests for acp_adapter.server — HermesACPAgent ACP server."""

import asyncio
import os
from types import SimpleNamespace
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

import acp
from acp.agent.router import build_agent_router
from acp.schema import (
    AgentCapabilities,
    AgentMessageChunk,
    AgentPlanUpdate,
    AgentThoughtChunk,
    AuthenticateResponse,
    AvailableCommandsUpdate,
    Implementation,
    InitializeResponse,
    LoadSessionResponse,
    NewSessionResponse,
    PromptResponse,
    ResumeSessionResponse,
    SessionModelState,
    SessionModeState,
    SetSessionConfigOptionResponse,
    SetSessionModelResponse,
    SetSessionModeResponse,
    SessionInfo,
    SessionInfoUpdate,
    TextContentBlock,
    ToolCallProgress,
    ToolCallStart,
    UsageUpdate,
    UserMessageChunk,
)
from acp_adapter.auth import TERMINAL_SETUP_AUTH_METHOD_ID
from acp_adapter.server import (
    ACP_MAX_MODELS_PER_PROVIDER,
    HermesACPAgent,
    HERMES_VERSION,
)
from acp_adapter.session import SessionManager
from hermes_state import SessionDB


@pytest.fixture()
def mock_manager():
    """SessionManager with a mock agent factory."""
    return SessionManager(agent_factory=lambda: MagicMock(name="MockAIAgent"))


@pytest.fixture()
def agent(mock_manager):
    """HermesACPAgent backed by a mock session manager."""
    return HermesACPAgent(session_manager=mock_manager)


@pytest.mark.asyncio
async def test_new_session_exposes_edit_approvals_as_modes_not_config_options(agent):
    resp = await agent.new_session(cwd="/tmp")

    assert resp.config_options is None
    assert isinstance(resp.modes, SessionModeState)
    assert resp.modes.current_mode_id == "default"
    assert [(mode.id, mode.name) for mode in resp.modes.available_modes] == [
        ("default", "Default"),
        ("accept_edits", "Accept Edits"),
        ("dont_ask", "Don't Ask"),
    ]


@pytest.mark.asyncio
async def test_set_config_option_persists_edit_approval_policy_without_advertising_config(agent):
    resp = await agent.new_session(cwd="/tmp")
    update = await agent.set_config_option(
        "edit_approval_policy",
        resp.session_id,
        "workspace_session",
    )
    state = agent.session_manager.get_session(resp.session_id)

    assert isinstance(update, SetSessionConfigOptionResponse)
    assert update.config_options == []
    assert getattr(state, "mode", None) == "accept_edits"


# ---------------------------------------------------------------------------
# initialize
# ---------------------------------------------------------------------------


class TestInitialize:
    @pytest.mark.asyncio
    async def test_initialize_returns_correct_protocol_version(self, agent):
        resp = await agent.initialize(protocol_version=1)
        assert isinstance(resp, InitializeResponse)
        assert resp.protocol_version == acp.PROTOCOL_VERSION




    @pytest.mark.asyncio
    async def test_initialize_advertises_provider_and_terminal_auth_methods(self, agent, monkeypatch):
        monkeypatch.setattr("acp_adapter.auth.detect_provider", lambda: "openrouter")
        monkeypatch.setattr("acp_adapter.server.detect_provider", lambda: "openrouter")

        resp = await agent.initialize(protocol_version=1)
        payloads = [method.model_dump(by_alias=True, exclude_none=True) for method in resp.auth_methods]

        assert payloads[0]["id"] == "openrouter"
        assert payloads[0]["name"] == "openrouter runtime credentials"
        terminal = next(payload for payload in payloads if payload["id"] == TERMINAL_SETUP_AUTH_METHOD_ID)
        assert terminal["type"] == "terminal"
        assert terminal["args"] == ["--setup"]



# ---------------------------------------------------------------------------
# authenticate
# ---------------------------------------------------------------------------


class TestAuthenticate:
    @pytest.mark.asyncio
    async def test_authenticate_with_matching_method_id(self, agent, monkeypatch):
        monkeypatch.setattr(
            "acp_adapter.server.detect_provider",
            lambda: "openrouter",
        )
        resp = await agent.authenticate(method_id="openrouter")
        assert isinstance(resp, AuthenticateResponse)

    @pytest.mark.asyncio
    async def test_authenticate_is_case_insensitive(self, agent, monkeypatch):
        monkeypatch.setattr(
            "acp_adapter.server.detect_provider",
            lambda: "openrouter",
        )
        resp = await agent.authenticate(method_id="OpenRouter")
        assert isinstance(resp, AuthenticateResponse)

    @pytest.mark.asyncio
    async def test_authenticate_rejects_mismatched_method_id(self, agent, monkeypatch):
        monkeypatch.setattr(
            "acp_adapter.server.detect_provider",
            lambda: "openrouter",
        )
        resp = await agent.authenticate(method_id="totally-invalid-method")
        assert resp is None

    @pytest.mark.asyncio
    async def test_authenticate_without_provider(self, agent, monkeypatch):
        monkeypatch.setattr(
            "acp_adapter.server.detect_provider",
            lambda: None,
        )
        resp = await agent.authenticate(method_id="openrouter")
        assert resp is None

    @pytest.mark.asyncio
    async def test_authenticate_accepts_terminal_setup_after_provider_configured(self, agent, monkeypatch):
        monkeypatch.setattr(
            "acp_adapter.server.detect_provider",
            lambda: "openrouter",
        )
        resp = await agent.authenticate(method_id=TERMINAL_SETUP_AUTH_METHOD_ID)
        assert isinstance(resp, AuthenticateResponse)



# ---------------------------------------------------------------------------
# new_session / cancel / load / resume
# ---------------------------------------------------------------------------


class TestSessionOps:

    @pytest.mark.asyncio
    async def test_new_session_returns_authenticated_cross_provider_model_state(self):
        manager = SessionManager(
            agent_factory=lambda: SimpleNamespace(
                model="gpt-5.4",
                provider="openai-codex",
                base_url="https://api.openai.com/v1",
            )
        )
        acp_agent = HermesACPAgent(session_manager=manager)
        picker_context = MagicMock()
        picker_context.with_overrides.return_value = picker_context
        payload = {
            "providers": [
                {
                    "slug": "anthropic",
                    "name": "Anthropic",
                    "models": ["claude-sonnet-4-6", "claude-sonnet-4-6"],
                },
                {
                    "slug": "openai-codex",
                    "name": "OpenAI Codex",
                    "models": [
                        {"id": "gpt-5.4"},
                        "gpt-5.4-mini",
                    ],
                },
            ],
        }

        with (
            patch("hermes_cli.inventory.load_picker_context", return_value=picker_context),
            patch("hermes_cli.inventory.build_models_payload", return_value=payload) as build_payload,
        ):
            resp = await acp_agent.new_session(cwd="/tmp")

        assert isinstance(resp.models, SessionModelState)
        assert resp.models.current_model_id == "openai-codex:gpt-5.4"
        assert [model.model_id for model in resp.models.available_models] == [
            "anthropic:claude-sonnet-4-6",
            "openai-codex:gpt-5.4",
            "openai-codex:gpt-5.4-mini",
        ]
        assert [model.name for model in resp.models.available_models] == [
            "Anthropic · claude-sonnet-4-6",
            "OpenAI Codex · gpt-5.4",
            "OpenAI Codex · gpt-5.4-mini",
        ]
        assert resp.models.available_models[1].description is not None
        assert "current" in resp.models.available_models[1].description
        picker_context.with_overrides.assert_called_once_with(
            current_provider="openai-codex",
            current_model="gpt-5.4",
            current_base_url="https://api.openai.com/v1",
        )
        build_payload.assert_called_once_with(
            picker_context,
            explicit_only=True,
            include_unconfigured=False,
            picker_hints=False,
            canonical_order=True,
            pricing=False,
            capabilities=False,
            refresh=False,
            probe_custom_providers=False,
            probe_current_custom_provider=False,
            max_models=ACP_MAX_MODELS_PER_PROVIDER,
        )



    @pytest.mark.asyncio
    async def test_available_commands_include_help(self, agent):
        help_cmd = next(
            (cmd for cmd in agent._available_commands() if cmd.name == "help"),
            None,
        )

        assert help_cmd is not None
        assert help_cmd.description == "List available commands"
        assert help_cmd.input is None


    def test_build_usage_update_for_zed_context_indicator(self, agent, mock_manager):
        state = mock_manager.create_session(cwd="/tmp")
        state.history = [{"role": "user", "content": "hello"}]
        state.agent.context_compressor = MagicMock(context_length=100_000)
        state.agent._cached_system_prompt = "system"
        state.agent.tools = [{"type": "function", "function": {"name": "demo"}}]

        with patch(
            "agent.model_metadata.estimate_request_tokens_rough",
            return_value=25_000,
        ):
            update = agent._build_usage_update(state)

        assert isinstance(update, UsageUpdate)
        assert update.session_update == "usage_update"
        assert update.size == 100_000
        assert update.used == 25_000




    @pytest.mark.asyncio
    async def test_load_session_not_found_returns_none(self, agent):
        resp = await agent.load_session(cwd="/tmp", session_id="bogus")
        assert resp is None






    @pytest.mark.asyncio
    async def test_resume_session_replays_persisted_history_to_client(self, agent):
        mock_conn = MagicMock(spec=acp.Client)
        mock_conn.session_update = AsyncMock()
        agent._conn = mock_conn

        new_resp = await agent.new_session(cwd="/tmp")
        state = agent.session_manager.get_session(new_resp.session_id)
        state.history = [{"role": "user", "content": "So tell me the current state"}]

        mock_conn.session_update.reset_mock()
        resp = await agent.resume_session(cwd="/tmp", session_id=new_resp.session_id)
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        assert isinstance(resp, ResumeSessionResponse)
        updates = [call.kwargs["update"] for call in mock_conn.session_update.await_args_list]
        assert any(
            isinstance(update, UserMessageChunk)
            and update.content.text == "So tell me the current state"
            for update in updates
        )











# ---------------------------------------------------------------------------
# list / fork
# ---------------------------------------------------------------------------


class TestListAndFork:
    @pytest.mark.asyncio
    async def test_fork_session(self, agent):
        new_resp = await agent.new_session(cwd="/original")
        fork_resp = await agent.fork_session(cwd="/forked", session_id=new_resp.session_id)
        assert fork_resp.session_id
        assert fork_resp.session_id != new_resp.session_id

    @pytest.mark.asyncio
    async def test_list_sessions_includes_title_and_updated_at(self, agent):
        with patch.object(
            agent.session_manager,
            "list_sessions",
            return_value=[
                {
                    "session_id": "session-1",
                    "cwd": "/tmp/project",
                    "title": "Fix Zed session history",
                    "updated_at": 123.0,
                }
            ],
        ):
            resp = await agent.list_sessions(cwd="/tmp/project")

        assert isinstance(resp.sessions[0], SessionInfo)
        assert resp.sessions[0].title == "Fix Zed session history"
        assert resp.sessions[0].updated_at == "123.0"






# ---------------------------------------------------------------------------
# session configuration / model routing
# ---------------------------------------------------------------------------


class TestSessionConfiguration:

    @pytest.mark.asyncio
    async def test_router_accepts_stable_session_config_methods(self, agent):
        new_resp = await agent.new_session(cwd="/tmp")
        router = build_agent_router(agent)

        mode_result = await router(
            "session/set_mode",
            {"modeId": "accept_edits", "sessionId": new_resp.session_id},
            False,
        )
        config_result = await router(
            "session/set_config_option",
            {
                "configId": "approval_mode",
                "sessionId": new_resp.session_id,
                "value": "auto",
            },
            False,
        )

        assert mode_result == {}
        assert config_result["configOptions"] == []





# ---------------------------------------------------------------------------
# prompt
# ---------------------------------------------------------------------------


class TestPrompt:
    @pytest.mark.asyncio
    async def test_prompt_returns_refusal_for_unknown_session(self, agent):
        prompt = [TextContentBlock(type="text", text="hello")]
        resp = await agent.prompt(prompt=prompt, session_id="nonexistent")
        assert isinstance(resp, PromptResponse)
        assert resp.stop_reason == "refusal"

    @pytest.mark.asyncio
    async def test_prompt_binds_session_id_into_subprocess_env(self, agent, mock_manager):
        """The ACP prompt path must bridge the session id into child subprocesses.

        Regression: ``set_session_vars`` was called with ``session_key`` only,
        leaving the ``HERMES_SESSION_ID`` ContextVar bound to the explicit ""
        default. Once the session-context machinery is engaged, that empty value
        is authoritative — so ``_make_run_env`` handed child subprocesses an
        empty ``HERMES_SESSION_ID`` instead of the session's own id.
        """
        from tools.environments.local import _make_run_env

        resp = await agent.new_session(cwd=".")
        state = mock_manager.get_session(resp.session_id)

        captured: dict[str, str | None] = {}

        def _run(*args, **kwargs):
            # Runs inside the session context copy set up by prompt().
            captured["child"] = _make_run_env({}).get("HERMES_SESSION_ID")
            return {"final_response": "ok", "messages": []}

        state.agent.run_conversation = _run
        state.agent.model = "test-model"
        state.agent.provider = "openrouter"

        mock_conn = MagicMock(spec=acp.Client)
        mock_conn.session_update = AsyncMock()
        agent._conn = mock_conn

        await agent.prompt(
            prompt=[TextContentBlock(type="text", text="hi")],
            session_id=resp.session_id,
        )

        assert captured.get("child") == resp.session_id












@pytest.mark.asyncio
async def test_model_switch_after_rotated_restore_keeps_live_tip_for_next_turn(
    agent, mock_manager, tmp_path
):
    import time

    db = SessionDB(tmp_path / "state.db")
    mock_manager._db_instance = db
    initial = mock_manager.create_session(cwd="/tmp")
    parent_id = initial.session_id
    child_id = f"{parent_id}-child"
    parent_messages = [
        {"role": "user", "content": "before compression"},
        {"role": "assistant", "content": "old reply"},
    ]
    db.append_messages_batch(parent_id, parent_messages)
    db.create_session(
        session_id=child_id,
        source="acp",
        parent_session_id=parent_id,
    )
    db.append_messages_batch(
        child_id,
        [{"role": "user", "content": "compressed summary"}],
    )

    def _end_parent(conn):
        conn.execute(
            "UPDATE sessions SET ended_at = ?, end_reason = 'compression' "
            "WHERE id = ?",
            (time.time(), parent_id),
        )

    db._execute_write(_end_parent)

    made_session_ids = []

    class FakeAgent:
        def __init__(self, session_id, model=None):
            self.session_id = session_id
            self.model = model or "test-model"
            self.provider = "openrouter"
            self.base_url = None
            self.api_mode = None
            self._session_db = db
            self._session_db_created = False

        def run_conversation(
            self,
            *,
            user_message,
            conversation_history,
            task_id,
            persist_user_message,
        ):
            made_session_ids.append(self.session_id)
            messages = list(conversation_history)
            messages.extend(
                [
                    {"role": "user", "content": persist_user_message},
                    {"role": "assistant", "content": "continued reply"},
                ]
            )
            return {"messages": messages, "final_response": "continued reply"}

    def _make_agent(**kwargs):
        return FakeAgent(
            session_id=kwargs["session_id"],
            model=kwargs.get("model"),
        )

    mock_manager._make_agent = _make_agent
    with mock_manager._lock:
        del mock_manager._sessions[parent_id]

    state = mock_manager.get_session(parent_id)
    assert state is not None
    assert state.agent.session_id == child_id

    with patch.object(
        agent,
        "_resolve_model_selection",
        return_value=("openrouter", "new-model"),
    ):
        response = await agent.set_session_model("new-model", parent_id)

    assert isinstance(response, SetSessionModelResponse)
    assert state.agent.session_id == child_id

    turn_response = await agent.prompt(
        prompt=[TextContentBlock(type="text", text="next turn")],
        session_id=parent_id,
    )

    assert turn_response.stop_reason == "end_turn"
    assert made_session_ids == [child_id]
    child_contents = [
        message.get("content")
        for message in db.get_messages_as_conversation(child_id)
    ]
    assert "compressed summary" in child_contents
    assert "next turn" in child_contents
    assert "continued reply" in child_contents




# ---------------------------------------------------------------------------
# on_connect
# ---------------------------------------------------------------------------


class TestOnConnect:
    def test_on_connect_stores_client(self, agent):
        mock_conn = MagicMock(spec=acp.Client)
        agent.on_connect(mock_conn)
        assert agent._conn is mock_conn


# ---------------------------------------------------------------------------
# Slash commands
# ---------------------------------------------------------------------------


class TestSlashCommands:
    """Test slash command dispatch in the ACP adapter."""

    def _make_state(self, mock_manager):
        state = mock_manager.create_session(cwd="/tmp")
        state.agent.model = "test-model"
        state.agent.provider = "openrouter"
        state.model = "test-model"
        return state

    def test_help_lists_commands(self, agent, mock_manager):
        state = self._make_state(mock_manager)
        result = agent._handle_slash_command("/help", state)
        assert result is not None
        assert "/help" in result
        assert "/model" in result
        assert "/tools" in result
        assert "/reset" in result

    def test_model_shows_current(self, agent, mock_manager):
        state = self._make_state(mock_manager)
        result = agent._handle_slash_command("/model", state)
        assert "test-model" in result





    def test_reset_clears_history(self, agent, mock_manager):
        state = self._make_state(mock_manager)
        state.history = [{"role": "user", "content": "hello"}]
        state.agent.session_id = state.session_id
        with patch.object(
            agent.session_manager, "save_session", wraps=agent.session_manager.save_session
        ) as mock_save:
            result = agent._handle_slash_command("/reset", state)
        assert "cleared" in result.lower()
        assert len(state.history) == 0
        mock_save.assert_called_once_with(state.session_id, sync_transcript=True)

    def test_reset_rejects_while_turn_running(self, agent, mock_manager):
        state = self._make_state(mock_manager)
        state.history = [{"role": "user", "content": "hello"}]
        state.is_running = True
        with patch.object(agent.session_manager, "save_session") as mock_save:
            result = agent._handle_slash_command("/reset", state)
        assert "while a turn is in progress" in result.lower()
        assert state.history == [{"role": "user", "content": "hello"}]
        mock_save.assert_not_called()

    def test_reset_restores_history_when_persist_fails(self, agent, mock_manager):
        state = self._make_state(mock_manager)
        state.history = [{"role": "user", "content": "keep me"}]
        state.agent.session_id = state.session_id
        with patch.object(
            agent.session_manager,
            "save_session",
            side_effect=RuntimeError("disk full"),
        ):
            result = agent._handle_slash_command("/reset", state)
        assert "failed to persist" in result.lower()
        assert state.history == [{"role": "user", "content": "keep me"}]

    def test_compact_compresses_context(self, agent, mock_manager):
        state = self._make_state(mock_manager)
        state.history = [
            {"role": "user", "content": "one"},
            {"role": "assistant", "content": "two"},
            {"role": "user", "content": "three"},
            {"role": "assistant", "content": "four"},
        ]
        state.agent.compression_enabled = True
        state.agent._cached_system_prompt = "system"
        state.agent.tools = None
        state.agent.session_id = state.session_id
        state.agent._session_db = None
        state.agent._session_db_created = False
        state.agent._last_compaction_in_place = False

        def _compress_context(messages, system_prompt, *, approx_tokens, task_id, force):
            # SessionDB must stay attached when present — mock has none.
            assert state.agent._session_db is None
            assert messages == [
                {"role": "user", "content": "one"},
                {"role": "assistant", "content": "two"},
                {"role": "user", "content": "three"},
                {"role": "assistant", "content": "four"},
            ]
            assert system_prompt == "system"
            assert approx_tokens == 40
            assert task_id == state.session_id
            assert force is True
            return [{"role": "user", "content": "summary"}], "new-system"

        state.agent._compress_context = MagicMock(side_effect=_compress_context)

        with (
            patch.object(
                agent.session_manager,
                "commit_compressed_transcript",
                wraps=agent.session_manager.commit_compressed_transcript,
            ) as mock_commit,
            patch(
                "agent.model_metadata.estimate_request_tokens_rough",
                side_effect=[40, 12],
            ),
        ):
            result = agent._handle_slash_command("/compress", state)

        assert "Context compressed: 4 -> 1 messages" in result
        assert "~40 -> ~12 tokens" in result
        assert len(state.history) == 1
        assert state.history[0]["role"] == "user"
        assert state.history[0]["content"] == "summary"
        assert state.agent._cached_system_prompt == "new-system"
        state.agent._compress_context.assert_called_once_with(
            [
                {"role": "user", "content": "one"},
                {"role": "assistant", "content": "two"},
                {"role": "user", "content": "three"},
                {"role": "assistant", "content": "four"},
            ],
            "system",
            approx_tokens=40,
            task_id=state.session_id,
            force=True,
        )
        assert mock_commit.call_count == 1
        commit_args, commit_kwargs = mock_commit.call_args
        assert commit_args[0] == state.session_id
        assert commit_args[1][0]["content"] == "summary"
        assert commit_kwargs["system_prompt"] == "new-system"
        db = agent.session_manager._get_db()
        live = db.get_messages_as_conversation(state.session_id)
        assert len(live) == 1
        assert live[0]["content"] == "summary"
        assert db.get_session(state.session_id)["system_prompt"] == "new-system"

    def test_compress_rejects_while_turn_running(self, agent, mock_manager):
        state = self._make_state(mock_manager)
        state.history = [{"role": "user", "content": "one"}]
        state.is_running = True
        state.agent._compress_context = MagicMock()
        result = agent._handle_slash_command("/compress", state)
        assert "while a turn is in progress" in result.lower()
        state.agent._compress_context.assert_not_called()

    def test_compress_restores_memory_when_persist_fails(self, agent, mock_manager):
        state = self._make_state(mock_manager)
        original = [
            {"role": "user", "content": "one"},
            {"role": "assistant", "content": "two"},
        ]
        state.history = list(original)
        state.agent._cached_system_prompt = "old-system"
        state.agent.tools = None
        state.agent.session_id = state.session_id
        state.agent._session_db = None
        state.agent._session_db_created = False
        state.agent._last_compaction_in_place = False
        state.agent._compress_context = MagicMock(
            return_value=([{"role": "user", "content": "summary"}], "new-system")
        )
        with (
            patch.object(
                agent.session_manager,
                "commit_compressed_transcript",
                side_effect=RuntimeError("locked"),
            ),
            patch(
                "agent.model_metadata.estimate_request_tokens_rough",
                return_value=10,
            ),
        ):
            result = agent._handle_slash_command("/compress", state)
        assert "failed to persist" in result.lower()
        assert state.history == original
        assert state.agent._cached_system_prompt == "old-system"

    def test_compress_rolls_back_db_and_memory_when_prompt_write_fails(
        self, agent, mock_manager, tmp_path
    ):
        db = SessionDB(tmp_path / "state.db")
        mock_manager._db_instance = db
        state = self._make_state(mock_manager)
        original = [
            {"role": "user", "content": "long question"},
            {"role": "assistant", "content": "long answer"},
        ]
        compacted = [{"role": "user", "content": "summary"}]
        db.append_messages_batch(state.session_id, original)
        db.update_system_prompt(state.session_id, "old-system")
        state.history = list(original)
        state.agent.session_id = state.session_id
        state.agent._session_db = None  # Exercise the manager-owned fallback commit.
        state.agent._session_db_created = False
        state.agent._cached_system_prompt = "old-system"
        state.agent.tools = None
        state.agent._last_compression_db_commit_succeeded = False
        state.agent._compress_context = MagicMock(
            return_value=(compacted, "new-system")
        )

        with (
            patch.object(
                db,
                "_store_system_prompt",
                side_effect=OSError("prompt storage failed"),
            ),
            patch(
                "agent.model_metadata.estimate_request_tokens_rough",
                return_value=10,
            ),
        ):
            result = agent._handle_slash_command("/compress", state)

        assert "failed to persist" in result.lower()
        assert state.history == original
        assert state.agent._cached_system_prompt == "old-system"
        assert [
            message.get("content")
            for message in db.get_messages_as_conversation(state.session_id)
        ] == ["long question", "long answer"]
        assert db.get_session(state.session_id)["system_prompt"] == "old-system"
        assert all(
            message.get("content") != "summary"
            for message in db.get_messages(
                state.session_id, include_compacted=True
            )
        )

    def test_compress_keeps_session_db_for_owned_in_place_commit(self, agent, mock_manager):
        state = self._make_state(mock_manager)
        db = agent.session_manager._get_db()
        state.history = [
            {"role": "user", "content": "long one"},
            {"role": "assistant", "content": "long two"},
        ]
        db.append_messages_batch(state.session_id, state.history)
        db.update_system_prompt(state.session_id, "old-system")
        state.agent.session_id = state.session_id
        state.agent._session_db = db
        state.agent._session_db_created = True
        state.agent._cached_system_prompt = "old-system"
        state.agent.tools = None
        state.agent._last_compaction_in_place = False

        def _compress_context(messages, system_prompt, *, approx_tokens, task_id, force):
            assert state.agent._session_db is db
            compacted = [{"role": "user", "content": "summary only"}]
            db.archive_and_compact(state.session_id, compacted)
            db.update_system_prompt(state.session_id, "new-system")
            state.agent._cached_system_prompt = "new-system"
            state.agent._last_compression_db_commit_succeeded = True
            state.agent._last_compaction_in_place = True
            return compacted, "new-system"

        state.agent._compress_context = MagicMock(side_effect=_compress_context)

        with (
            patch.object(agent.session_manager, "save_session") as mock_save,
            patch.object(
                agent.session_manager, "commit_compressed_transcript"
            ) as mock_commit,
            patch(
                "agent.model_metadata.estimate_request_tokens_rough",
                side_effect=[40, 12],
            ),
        ):
            result = agent._handle_slash_command("/compress", state)

        assert "Context compressed: 2 -> 1 messages" in result
        assert len(state.history) == 1
        assert state.history[0]["content"] == "summary only"
        mock_save.assert_not_called()
        mock_commit.assert_not_called()
        live = db.get_messages_as_conversation(state.session_id)
        assert len(live) == 1
        assert live[0]["content"] == "summary only"
        assert db.get_session(state.session_id)["system_prompt"] == "new-system"
        archived = db.get_messages(state.session_id, include_compacted=True)
        assert any(row.get("content") == "long one" for row in archived)



    def test_compress_after_cold_restore_uses_commit_signal_and_preserves_tail(
        self, agent, mock_manager, tmp_path
    ):
        db = SessionDB(tmp_path / "state.db")
        mock_manager._db_instance = db
        initial = self._make_state(mock_manager)
        session_id = initial.session_id
        original = [
            {"role": "user", "content": "long question"},
            {"role": "assistant", "content": "long answer"},
        ]
        initial.history = list(original)
        db.append_messages_batch(session_id, original)
        db.update_system_prompt(session_id, "old-system")

        with mock_manager._lock:
            del mock_manager._sessions[session_id]
        mock_manager._agent_factory = lambda: MagicMock(
            name="RestoredAIAgent",
            session_id=session_id,
            _session_db=db,
            _session_db_created=False,
            _cached_system_prompt="old-system",
            tools=None,
        )
        state = mock_manager.get_session(session_id)
        assert state is not None
        assert state.agent._session_db_created is False
        assert state.agent._session_db is db

        compacted = [{"role": "user", "content": "summary"}]
        concurrent_tail = [
            {"role": "user", "content": "concurrent user turn"},
            {"role": "assistant", "content": "concurrent assistant reply"},
        ]

        def _compress_context(
            messages, system_prompt, *, approx_tokens, task_id, force
        ):
            watermark = db.get_active_message_watermark(session_id)
            db.append_messages_batch(session_id, concurrent_tail)
            db.archive_and_compact(
                session_id,
                compacted,
                watermark=watermark,
            )
            db.update_system_prompt(session_id, "new-system")
            state.agent._last_compression_db_commit_succeeded = True
            state.agent._cached_system_prompt = "new-system"
            return compacted, "new-system"

        state.agent._compress_context = MagicMock(side_effect=_compress_context)

        with (
            patch.object(
                mock_manager,
                "commit_compressed_transcript",
                wraps=mock_manager.commit_compressed_transcript,
            ) as mock_commit,
            patch(
                "agent.model_metadata.estimate_request_tokens_rough",
                side_effect=[40, 12],
            ),
        ):
            result = agent._handle_slash_command("/compress", state)

        assert "Context compressed: 2 -> 1 messages" in result
        mock_commit.assert_not_called()
        live_contents = [
            message.get("content")
            for message in db.get_messages_as_conversation(session_id)
        ]
        assert live_contents == [
            "summary",
            "concurrent user turn",
            "concurrent assistant reply",
        ]
        archived_contents = [
            message.get("content")
            for message in db.get_messages(session_id, include_compacted=True)
        ]
        assert "long question" in archived_contents
        assert "long answer" in archived_contents


    def test_unknown_command_returns_none(self, agent, mock_manager):
        state = self._make_state(mock_manager)
        result = agent._handle_slash_command("/nonexistent", state)
        assert result is None


    def test_slash_handler_cwd_pin_does_not_leak(self, agent, mock_manager, tmp_path):
        """The pin is scoped to the handler's own context copy.

        Concurrent ACP sessions share the event loop, so a handler that pinned
        the ambient context would leave its workspace bound for whatever runs
        next. Asserting the ambient value is unchanged after dispatch keeps the
        fix from trading one cross-session leak for another.
        """
        from agent.runtime_cwd import resolve_agent_cwd

        workspace = tmp_path / "project"
        workspace.mkdir()
        state = mock_manager.create_session(cwd=str(workspace))
        state.cwd = str(workspace)
        state.agent.model = "test-model"
        state.agent.provider = "openrouter"

        before = str(resolve_agent_cwd())
        agent._handle_slash_command("/help", state)
        assert str(resolve_agent_cwd()) == before





# ---------------------------------------------------------------------------
# _register_session_mcp_servers
# ---------------------------------------------------------------------------


class TestRegisterSessionMcpServers:
    """Tests for ACP MCP server registration in session lifecycle."""

    @pytest.mark.asyncio
    async def test_noop_when_no_servers(self, agent, mock_manager):
        """No-op when mcp_servers is None or empty."""
        state = mock_manager.create_session(cwd="/tmp")
        # Should not raise
        await agent._register_session_mcp_servers(state, None)
        await agent._register_session_mcp_servers(state, [])

    @pytest.mark.asyncio
    async def test_registers_stdio_servers(self, agent, mock_manager):
        """McpServerStdio servers are converted and passed to register_mcp_servers."""
        from acp.schema import McpServerStdio, EnvVariable

        state = mock_manager.create_session(cwd="/tmp")
        # Give the mock agent the attributes _register_session_mcp_servers reads
        state.agent.enabled_toolsets = ["hermes-acp"]
        state.agent.disabled_toolsets = None
        state.agent.tools = []
        state.agent.valid_tool_names = set()

        server = McpServerStdio(
            name="test-server",
            command="/usr/bin/test",
            args=["--flag"],
            env=[EnvVariable(name="KEY", value="val")],
        )

        registered_config = {}
        def capture_register(config_map):
            registered_config.update(config_map)
            return ["mcp_test_server_tool1"]

        with patch("tools.mcp_tool.register_mcp_servers", side_effect=capture_register), \
             patch("model_tools.get_tool_definitions", return_value=[]):
            await agent._register_session_mcp_servers(state, [server])

        assert "test-server" in registered_config
        cfg = registered_config["test-server"]
        assert cfg["command"] == "/usr/bin/test"
        assert cfg["args"] == ["--flag"]
        assert cfg["env"] == {"KEY": "val"}


    @pytest.mark.asyncio
    async def test_refreshes_agent_tool_surface(self, agent, mock_manager):
        """After MCP registration, agent.tools and valid_tool_names are refreshed."""
        from acp.schema import McpServerStdio

        state = mock_manager.create_session(cwd="/tmp")
        state.agent.enabled_toolsets = ["hermes-acp"]
        state.agent.disabled_toolsets = None
        state.agent.tools = []
        state.agent.valid_tool_names = set()
        state.agent._cached_system_prompt = "old prompt"
        state.agent._memory_manager = SimpleNamespace(
            get_all_tool_schemas=lambda: [
                {"name": "hindsight_recall", "description": "Recall", "parameters": {}}
            ]
        )

        server = McpServerStdio(
            name="srv",
            command="/bin/test",
            args=[],
            env=[],
        )

        fake_tools = [
            {"function": {"name": "mcp_srv_search"}},
            {"function": {"name": "memory"}},
            {"function": {"name": "terminal"}},
        ]

        with patch("tools.mcp_tool.register_mcp_servers", return_value=["mcp_srv_search"]), \
             patch("model_tools.get_tool_definitions", return_value=fake_tools) as mock_defs:
            await agent._register_session_mcp_servers(state, [server])

        mock_defs.assert_called_once_with(
            enabled_toolsets=["hermes-acp", "mcp-srv"],
            disabled_toolsets=None,
            quiet_mode=True,
        )
        assert state.agent.enabled_toolsets == ["hermes-acp", "mcp-srv"]
        assert state.agent.tools is fake_tools
        assert state.agent.tools[-1] == {
            "type": "function",
            "function": {
                "name": "hindsight_recall",
                "description": "Recall",
                "parameters": {},
            },
        }
        assert state.agent.valid_tool_names == {
            "hindsight_recall",
            "memory",
            "mcp_srv_search",
            "terminal",
        }
        # _invalidate_system_prompt should have been called
        state.agent._invalidate_system_prompt.assert_called_once()

    @pytest.mark.asyncio
    async def test_register_failure_logs_warning(self, agent, mock_manager):
        """If register_mcp_servers raises, warning is logged but no crash."""
        from acp.schema import McpServerStdio

        state = mock_manager.create_session(cwd="/tmp")
        server = McpServerStdio(
            name="bad",
            command="/nonexistent",
            args=[],
            env=[],
        )

        with patch("tools.mcp_tool.register_mcp_servers", side_effect=RuntimeError("boom")):
            # Should not raise
            await agent._register_session_mcp_servers(state, [server])
