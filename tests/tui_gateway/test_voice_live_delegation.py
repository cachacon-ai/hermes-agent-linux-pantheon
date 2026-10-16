"""GPT-Live voice chat mode: the full-duplex voice frontend that delegates to Hermes.

The live voice model owns the microphone and speaker and has no tools; every real
request is delegated to Hermes as a normal turn on the open session. Three
contracts matter and are pinned here:

* the gateway never hands the OpenAI key to the renderer — ``POST /v1/live/sessions``
  is performed server-side from the renderer's SDP offer, with the session pinned to
  client delegation so Hermes (any model) is the backend;
* a turn submitted from the live voice surface carries the spoken-delegation note on
  the MODEL INPUT only (the byte-stable system prompt is untouched), exactly like the
  HUD note it sits beside;
* the transcript is PER-TURN data. It is admitted with the submit that carried it and
  threaded into that one turn, never parked on session state where an overlapping or
  queued submit could inherit it.

Fork divergence from upstream: a ``voice_context`` arriving on any surface other than
``voice-live`` is REFUSED (invalid params) instead of being silently dropped, and
context longer than the 6000-char limit is refused rather than clipped.
"""

import io
import json
import logging
import threading
import types
import urllib.error

import pytest

from gateway.session_surface import VOICE_CONTEXT_LIMIT, surface_note
from tools import voice_live
from tui_gateway import server


def _session(**extra):
    return {
        "agent": types.SimpleNamespace(valid_tool_names=set()),
        "session_key": "session-key",
        "history": [],
        "history_lock": threading.Lock(),
        "history_version": 0,
        "running": True,
        "transport": None,
        "attached_images": [],
        **extra,
    }


class TestSessionCreation:
    ANSWER_SDP = "v=0\r\no=- 7 2 IN IP4 127.0.0.1\r\ns=-\r\nt=0 0\r\na=group:BUNDLE 0\r\n"

    @staticmethod
    def _gpt_live(monkeypatch, live=None):
        """This mode is opt-in: the vendor functions refuse unless the profile selected it."""
        monkeypatch.setattr(
            voice_live, "_voice_section", lambda: {"voice_chat_mode": "gpt-live"}
        )
        if live is not None:
            monkeypatch.setattr(voice_live, "_live_section", lambda voice=None: live)

    def test_client_delegation_and_key_stay_server_side(self, monkeypatch):
        """Whatever the renderer sends, the vendor request pins ``delegation.type == client``
        (Hermes is the backend) and authenticates with the resolved key; the client only ever
        sees the vendor answer."""
        captured = {}

        class _Resp:
            def __init__(self, body):
                self._body = body

            def read(self):
                return self._body

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        def fake_urlopen(req, timeout=0):
            captured["url"] = req.full_url
            captured["auth"] = req.get_header("Authorization")
            captured["body"] = json.loads(req.data)
            return _Resp(
                json.dumps(
                    {
                        "session": {"id": "live_x"},
                        "transport": {"type": "webrtc", "sdp": self.ANSWER_SDP},
                    }
                ).encode()
            )

        monkeypatch.setattr(voice_live.urllib.request, "urlopen", fake_urlopen)
        self._gpt_live(
            monkeypatch, {"voice": "willow", "instructions": "Speak Spanish."}
        )
        monkeypatch.setattr(
            voice_live, "_resolve_credentials", lambda live: ("sk-test", "https://api.example/v1")
        )

        result = voice_live.create_webrtc_session(
            "v=0 offer", history=[{"type": "message", "role": "user", "content": "hello"}]
        )

        assert result["transport"]["sdp"] == self.ANSWER_SDP
        assert captured["url"] == "https://api.example/v1/live/sessions"
        assert captured["auth"] == "Bearer sk-test"
        session = captured["body"]["session"]
        assert session["delegation"] == {"type": "client"}
        assert session["audio"]["output"]["voice"] == "willow"
        assert session["instructions"].endswith("Speak Spanish.")
        # History is coerced to one text part per message (R4).
        assert session["input"] == [
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hello"}]}
        ]
        assert captured["body"]["transport"] == {"type": "webrtc", "sdp": "v=0 offer"}
        assert "sk-test" not in json.dumps(result)

    def test_missing_key_refuses_before_any_network(self, monkeypatch):
        self._gpt_live(monkeypatch)
        monkeypatch.setattr(
            voice_live,
            "_resolve_credentials",
            lambda live: ("", voice_live.DEFAULT_LIVE_BASE_URL),
        )
        monkeypatch.setattr(
            voice_live.urllib.request, "urlopen", lambda *a, **k: pytest.fail("must not call the vendor")
        )

        with pytest.raises(voice_live.VoiceLiveMissingKeyError):
            voice_live.create_webrtc_session("v=0 offer")
        assert voice_live.resolve_gpt_live_status()["available"] is False

    def test_sdp_offer_is_forwarded_byte_exact(self, monkeypatch):
        """A stripped offer answers 400 "failed to unmarshal SDP: EOF" at the vendor — the
        offer goes over the wire exactly as produced, trailing CRLF included."""
        captured = {}

        class _Resp:
            def read(self):
                return json.dumps(
                    {"session": {"id": "live_x"}, "transport": {"sdp": "v=0\r\n"}}
                ).encode()

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        def fake_urlopen(req, timeout=0):
            captured["sdp"] = json.loads(req.data)["transport"]["sdp"]
            return _Resp()

        monkeypatch.setattr(voice_live.urllib.request, "urlopen", fake_urlopen)
        self._gpt_live(monkeypatch)
        monkeypatch.setattr(voice_live, "_resolve_credentials", lambda live: ("sk-test", "https://x/v1"))
        offer = "v=0\r\no=- 1 1 IN IP4 127.0.0.1\r\ns=-\r\n"

        voice_live.create_webrtc_session(offer)

        assert captured["sdp"] == offer


class TestModeGate:
    """R1: the mode is the opt-in. A resolving key does not enable GPT-Live."""

    @pytest.mark.parametrize("mode", ["chained", "Chained", "gpt_live_typo", "", None])
    def test_a_non_live_mode_refuses_before_the_vendor(self, monkeypatch, mode):
        monkeypatch.setattr(
            voice_live, "_voice_section", lambda: {"voice_chat_mode": mode}
        )
        monkeypatch.setattr(voice_live, "_live_section", lambda voice=None: {})
        monkeypatch.setattr(
            voice_live, "_resolve_credentials", lambda live: ("sk-test", "https://x/v1")
        )
        monkeypatch.setattr(
            voice_live.urllib.request,
            "urlopen",
            lambda *a, **k: pytest.fail("the vendor must not be called"),
        )

        with pytest.raises(voice_live.VoiceLiveDisabledError):
            voice_live.create_webrtc_session("v=0 offer")

    def test_a_configured_key_does_not_enable_the_mode(self, monkeypatch):
        monkeypatch.setattr(voice_live, "_voice_section", lambda: {})
        monkeypatch.setattr(
            voice_live, "_resolve_credentials", lambda live: ("sk-test", "https://x/v1")
        )

        assert voice_live.resolve_gpt_live_status()["available"] is True
        assert voice_live.resolve_gpt_live_status()["mode"] == "chained"

    @pytest.mark.parametrize("spelling", ["gpt-live", "gpt_live", "GPT-Live", "live", "gptlive"])
    def test_the_enabled_spellings_all_pass(self, monkeypatch, spelling):
        monkeypatch.setattr(
            voice_live, "_voice_section", lambda: {"voice_chat_mode": spelling}
        )
        assert voice_live.ensure_gpt_live_mode() == "gpt-live"


class TestVendorAnswerValidation:
    """R3: a 200 that is not a usable session answer is a failed session, not a success."""

    def _install(self, monkeypatch, payload):
        monkeypatch.setattr(
            voice_live, "_voice_section", lambda: {"voice_chat_mode": "gpt-live"}
        )
        monkeypatch.setattr(
            voice_live, "_resolve_credentials", lambda live: ("sk-test", "https://x/v1")
        )

        class _Resp:
            def read(self):
                return json.dumps(payload).encode()

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        monkeypatch.setattr(
            voice_live.urllib.request, "urlopen", lambda *a, **k: _Resp()
        )

    @pytest.mark.parametrize(
        "payload",
        [
            pytest.param({}, id="empty-object"),
            pytest.param({"session": {}, "transport": {"type": "webrtc"}}, id="no-sdp"),
            pytest.param({"session": {"id": "x"}}, id="no-transport"),
            pytest.param({"session": {"id": ""}, "transport": {"sdp": "v=0\r\n"}}, id="blank-id"),
            pytest.param({"session": {"id": 7}, "transport": {"sdp": "v=0\r\n"}}, id="non-string-id"),
            pytest.param({"session": {"id": "x"}, "transport": {"sdp": ""}}, id="empty-sdp"),
            pytest.param({"session": {"id": "x"}, "transport": {"sdp": 5}}, id="non-string-sdp"),
            pytest.param(
                {"session": {"id": "x"}, "transport": {"sdp": "<html>oops</html>"}},
                id="sdp-is-not-an-sdp",
            ),
            pytest.param(
                {"session": "not-an-object", "transport": {"sdp": "v=0\r\n"}},
                id="session-not-an-object",
            ),
            pytest.param([1, 2, 3], id="answer-is-a-list"),
        ],
    )
    def test_an_unusable_answer_is_a_typed_protocol_error(self, monkeypatch, payload):
        self._install(monkeypatch, payload)

        with pytest.raises(voice_live.VoiceLiveProtocolError):
            voice_live.create_webrtc_session("v=0 offer")

    def test_a_well_formed_answer_passes_through(self, monkeypatch):
        payload = {"session": {"id": "live_x"}, "transport": {"type": "webrtc", "sdp": "v=0\r\n"}}
        self._install(monkeypatch, payload)

        assert voice_live.create_webrtc_session("v=0 offer") == payload


class TestSessionClose:
    """Vendor hangup stays server-side; bodies and keys never leak."""

    def test_hangup_posts_to_the_vendor_endpoint(self, monkeypatch):
        captured = {}

        class _Resp:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        def fake_urlopen(req, timeout=0):
            captured["url"] = req.full_url
            captured["method"] = req.get_method()
            return _Resp()

        monkeypatch.setattr(
            voice_live, "_voice_section", lambda: {"voice_chat_mode": "gpt-live"}
        )
        monkeypatch.setattr(
            voice_live, "_resolve_credentials", lambda live: ("sk-test", "https://api.example/v1")
        )
        monkeypatch.setattr(voice_live.urllib.request, "urlopen", fake_urlopen)

        result = voice_live.close_live_session("live_sess_abc")

        assert result == {"closed": True, "method": "hangup"}
        assert captured["method"] == "POST"
        assert captured["url"] == "https://api.example/v1/live/sessions/live_sess_abc/hangup"

    def test_hangup_rejection_never_logs_the_body(self, monkeypatch, caplog):
        secret = "sk-live-SECRET-do-not-leak-abcdef"
        monkeypatch.setattr(
            voice_live, "_voice_section", lambda: {"voice_chat_mode": "gpt-live"}
        )
        monkeypatch.setattr(
            voice_live, "_resolve_credentials", lambda live: (secret, "https://x/v1")
        )

        def raise_http_error(req, timeout=0):
            raise urllib.error.HTTPError(
                req.full_url,
                404,
                "Not Found",
                {},
                io.BytesIO(json.dumps({"error": {"message": secret}}).encode()),
            )

        monkeypatch.setattr(voice_live.urllib.request, "urlopen", raise_http_error)

        with caplog.at_level(logging.DEBUG):
            with pytest.raises(voice_live.VoiceLiveUpstreamError) as excinfo:
                voice_live.close_live_session("live_gone")

        logged = "\n".join(record.getMessage() for record in caplog.records)
        assert secret not in logged
        assert secret not in str(excinfo.value)
        assert "404" in logged and "not_found" in logged


class TestVendorErrorHygiene:
    """R6: the upstream body is untrusted and can echo key material. Never logged, never raised."""

    def test_the_rejected_body_is_not_logged_or_raised(self, monkeypatch, caplog):
        secret = "sk-live-SECRET-do-not-leak-abcdef"
        monkeypatch.setattr(
            voice_live, "_voice_section", lambda: {"voice_chat_mode": "gpt-live"}
        )
        monkeypatch.setattr(
            voice_live, "_resolve_credentials", lambda live: (secret, "https://x/v1")
        )

        def raise_http_error(req, timeout=0):
            raise urllib.error.HTTPError(
                req.full_url,
                401,
                "Unauthorized",
                {},
                io.BytesIO(
                    json.dumps(
                        {"error": {"message": f"incorrect api key provided: {secret}"}}
                    ).encode()
                ),
            )

        monkeypatch.setattr(voice_live.urllib.request, "urlopen", raise_http_error)

        with caplog.at_level(logging.DEBUG):
            with pytest.raises(voice_live.VoiceLiveUpstreamError) as excinfo:
                voice_live.create_webrtc_session("v=0 offer")

        logged = "\n".join(record.getMessage() for record in caplog.records)
        assert secret not in logged
        assert secret not in str(excinfo.value)
        # The operator still gets what they need: status and category.
        assert "401" in str(excinfo.value)
        assert "auth" in str(excinfo.value)
        assert "401" in logged and "auth" in logged


class TestSessionHistoryContract:
    """R4: the live conversation API takes <=128 messages of text, ~8192 combined tokens."""

    @staticmethod
    def _msg(role, text, index=0):
        return {"type": "message", "role": role, "content": [{"type": "input_text", "text": text}]}

    def test_a_valid_history_is_coerced_to_one_text_part_each(self):
        result = voice_live.normalize_session_history(
            [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
        )

        assert result == [
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]},
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "input_text", "text": "hello"}],
            },
        ]

    def test_multiple_text_parts_are_joined_into_one(self):
        """The API allows one text part per message; extra parts must not be forwarded as-is."""
        result = voice_live.normalize_session_history(
            [{"role": "user", "content": [{"type": "input_text", "text": "a"},
                                          {"type": "text", "text": "b"}]}]
        )

        assert result[0]["content"] == [{"type": "input_text", "text": "a\nb"}]

    @pytest.mark.parametrize(
        "message",
        [
            pytest.param({"role": "tool", "content": "x"}, id="disallowed-role"),
            pytest.param({"role": "developer", "content": "x"}, id="developer-role"),
            pytest.param({"content": "x"}, id="missing-role"),
            pytest.param({"role": 3, "content": "x"}, id="non-string-role"),
            pytest.param({"role": "user", "content": 5}, id="non-string-content"),
            pytest.param({"role": "user", "content": None}, id="null-content"),
            pytest.param(
                {"role": "user", "content": [{"type": "input_audio", "data": "x"}]},
                id="audio-part",
            ),
            pytest.param(
                {"role": "user", "content": [{"type": "text", "text": {"a": 1}}]},
                id="non-string-text",
            ),
            pytest.param({"role": "user", "content": [5]}, id="non-object-part"),
            pytest.param("a string", id="message-not-an-object"),
        ],
    )
    def test_a_structurally_invalid_entry_is_refused_not_dropped(self, message):
        with pytest.raises(voice_live.VoiceLiveHistoryError):
            voice_live.normalize_session_history([message])

    def test_a_history_that_is_not_a_list_is_refused(self):
        with pytest.raises(voice_live.VoiceLiveHistoryError):
            voice_live.normalize_session_history({"role": "user"})

    def test_history_is_capped_at_the_message_limit_oldest_first(self):
        history = [self._msg("user", f"m{i}") for i in range(200)]

        result = voice_live.normalize_session_history(history)

        assert len(result) == voice_live.LIVE_HISTORY_MAX_MESSAGES
        # Newest survives, oldest is what gets dropped.
        assert result[-1]["content"][0]["text"] == "m199"
        assert result[0]["content"][0]["text"] == "m72"

    def test_history_is_pruned_to_the_token_budget_oldest_first(self):
        filler = "x" * 4000  # ~1000 tokens per message at the conservative estimate
        history = [self._msg("user", filler) for _ in range(10)]

        result = voice_live.normalize_session_history(history)

        budget_chars = voice_live.LIVE_HISTORY_MAX_TOKENS * voice_live.LIVE_HISTORY_CHARS_PER_TOKEN
        kept_chars = sum(len(m["content"][0]["text"]) for m in result)
        assert kept_chars <= budget_chars
        # The newest turn is never the one dropped.
        assert len(result) < 10 and result[-1]["content"][0]["text"] == filler

    def test_an_oversized_single_message_is_kept_rather_than_dropped(self):
        """Pruning must not silently delete the only turn: the client asked to resume it."""
        history = [self._msg("user", "y" * (200_000))]

        result = voice_live.normalize_session_history(history)

        assert len(result) == 1
        assert result[0]["content"][0]["text"].startswith("y")

    def test_empty_messages_are_dropped_before_the_budget_is_spent(self):
        history = [self._msg("user", "  "), self._msg("user", "real")]

        assert voice_live.normalize_session_history(history) == [
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "real"}]}
        ]

    def test_no_history_means_no_input_key(self):
        assert "input" not in voice_live.build_session_config(None)
        assert "input" not in voice_live.build_session_config([])

    def test_the_pruned_history_is_what_reaches_the_vendor(self, monkeypatch):
        """The contract is enforced on the wire, not only in the helper."""
        captured = {}

        class _Resp:
            def read(self):
                return json.dumps(
                    {"session": {"id": "live_x"}, "transport": {"sdp": "v=0\r\n"}}
                ).encode()

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        def fake_urlopen(req, timeout=0):
            captured["body"] = json.loads(req.data)
            return _Resp()

        monkeypatch.setattr(voice_live.urllib.request, "urlopen", fake_urlopen)
        monkeypatch.setattr(
            voice_live, "_voice_section", lambda: {"voice_chat_mode": "gpt-live"}
        )
        monkeypatch.setattr(
            voice_live, "_resolve_credentials", lambda live: ("sk-test", "https://x/v1")
        )

        with pytest.raises(voice_live.VoiceLiveHistoryError):
            voice_live.create_webrtc_session(
                "v=0", history=[{"role": "tool", "content": "x"}]
            )

        assert "body" not in captured  # refused before the wire, not after


@pytest.fixture
def busy_session():
    # A running session takes the busy path, which returns before any of the
    # agent/DB machinery — enough to observe what submit admitted.
    session = _session()
    server._sessions["sid"] = session
    yield session
    server._sessions.pop("sid", None)


def _submit(**params):
    return server._methods["prompt.submit"](
        "r1", {"session_id": "sid", "text": "what's the weather", "queued": True, **params}
    )


class TestVoiceLiveTurnNote:
    def test_live_surface_queued_keeps_its_own_transcript(self, busy_session):
        """The persisted row is the user's words; the transcript rides the queued turn."""
        _submit(
            surface="voice-live",
            voice_context="Voice assistant: Hi\nUser: what's the weather",
        )

        assert busy_session["client_surface"] == "voice-live"
        committed = busy_session["queued_prompt"]["voice_live_turn"]
        assert committed == {
            "surface": "voice-live",
            "voice_context": "Voice assistant: Hi\nUser: what's the weather",
        }
        note = surface_note(committed)
        assert note.startswith(voice_live.VOICE_LIVE_TURN_NOTE)
        assert "User: what's the weather" in note

    def test_live_surface_without_a_transcript_still_gets_the_contract(self, busy_session):
        _submit(surface="voice-live")

        committed = busy_session["queued_prompt"]["voice_live_turn"]
        assert committed == {"surface": "voice-live"}
        assert surface_note(committed) == voice_live.VOICE_LIVE_TURN_NOTE

    def test_voice_context_is_refused_off_the_live_surface(self, busy_session):
        """Deliberate fork divergence: upstream ignores it. The transcript is model input, so
        a client that can attach one to an arbitrary surface could put words in a turn the
        user never dictated."""
        response = _submit(voice_context="User: smuggled")

        assert response["error"]["code"] == 4004
        assert "voice-live" in response["error"]["message"]
        assert not busy_session.get("queued_prompt")
        assert busy_session.get("client_surface") != "voice-live"

    def test_voice_context_is_refused_on_the_hud_surface(self, busy_session):
        response = _submit(surface="hud", voice_context="User: smuggled")

        assert response["error"]["code"] == 4004
        assert not busy_session.get("queued_prompt")

    def test_voice_context_on_an_unknown_surface_is_refused(self, busy_session):
        response = _submit(surface="pet-overlay", voice_context="User: smuggled")

        assert response["error"]["code"] == 4004
        assert not busy_session.get("queued_prompt")

    def test_over_long_voice_context_is_refused_not_clipped(self, busy_session):
        response = _submit(surface="voice-live", voice_context="x" * (VOICE_CONTEXT_LIMIT + 1))

        assert response["error"]["code"] == 4004
        assert not busy_session.get("queued_prompt")

    def test_non_string_voice_context_is_refused(self, busy_session):
        response = _submit(surface="voice-live", voice_context={"role": "user"})

        assert response["error"]["code"] == 4004
        assert not busy_session.get("queued_prompt")

    def test_context_at_the_limit_is_accepted(self, busy_session):
        context = "x" * VOICE_CONTEXT_LIMIT
        response = _submit(surface="voice-live", voice_context=context)

        assert "error" not in response
        assert busy_session["queued_prompt"]["voice_live_turn"]["voice_context"] == context

    def test_plain_window_submit_clears_the_live_surface(self, busy_session):
        _submit(surface="voice-live")
        server._methods["prompt.submit"](
            "r2", {"session_id": "sid", "text": "y", "queued": True}
        )

        assert busy_session["client_surface"] == ""
        # The live delegation keeps its own envelope; the plain turn behind it
        # must not inherit the delegation's surface.
        assert busy_session["queued_prompt"]["voice_live_turn"] == {"surface": "voice-live"}
        assert busy_session["queued_prompts"][0]["text"] == "y"
        assert busy_session["queued_prompts"][0].get("voice_live_turn") is None

    def test_overlapping_voice_turns_keep_their_own_transcripts(self, busy_session):
        """Two live delegations queue behind one another. Merging them would run the second
        turn's words against the first turn's transcript."""
        _submit(surface="voice-live", voice_context="User: first")
        _submit(surface="voice-live", voice_context="User: second")

        queued = busy_session["queued_prompt"]
        followups = busy_session["queued_prompts"]
        assert queued["voice_live_turn"]["voice_context"] == "User: first"
        assert followups[0]["voice_live_turn"]["voice_context"] == "User: second"

    def test_a_voice_turn_is_not_merged_into_a_plain_queued_prompt(self, busy_session):
        _submit(text="typed while busy")
        _submit(surface="voice-live", voice_context="User: spoken")

        assert busy_session["queued_prompt"].get("voice_live_turn") is None
        assert busy_session["queued_prompts"][0]["voice_live_turn"] == {
            "surface": "voice-live",
            "voice_context": "User: spoken",
        }


class TestPromptCacheSafety:
    """The delegation note rides the model input. The system prompt has to stay
    byte-stable for the life of the conversation, or the cached prefix that makes a
    long voice conversation affordable is gone."""

    def test_the_note_is_prepended_to_the_model_input_only(self):
        committed = {"surface": "voice-live", "voice_context": "User: hi"}
        note = surface_note(committed)

        model_input = server._prepend_note("what's the weather", note)

        assert model_input.startswith(voice_live.VOICE_LIVE_TURN_NOTE)
        assert model_input.endswith("what's the weather")
        # What the user actually said stays clean, so the persisted row is their words.
        assert "what's the weather" not in note

    def test_a_live_turn_persists_no_scaffolding(self, busy_session):
        before = json.dumps(busy_session["history"])

        _submit(surface="voice-live", voice_context="User: hi")

        assert json.dumps(busy_session["history"]) == before

    def test_the_surface_never_lands_in_session_state(self, busy_session):
        """Only the HUD's existing surface marker is mirrored. The transcript lives on the
        turn it belongs to."""
        _submit(surface="voice-live", voice_context="User: hi")

        assert busy_session["client_surface"] == "voice-live"
        assert "voice_live_context" not in busy_session


class TestComputeHostPropagation:
    """An isolated worker runs the same turn prologue, so the delegation must travel
    with the frame — or the turn runs without it and nobody notices."""

    def _frame(self, **kwargs):
        session = _session(running=False)
        return server._compute_host_turn_frame("r1", "sid", session, "text", **kwargs)

    def test_the_frame_carries_the_committed_turn(self):
        committed = {"surface": "voice-live", "voice_context": "User: hi"}

        frame = self._frame(voice_live_turn=committed)

        assert frame["voice_live_turn"] == committed

    def test_a_plain_turn_carries_no_surface_field(self):
        assert "voice_live_turn" not in self._frame()

    def test_the_worker_accepts_a_delegation(self):
        from tui_gateway import compute_host

        assert compute_host._worker_carries_voice_live(server) is True

    def test_a_worker_without_the_turn_parameter_fails_closed(self):
        """A stale worker would accept the field and drop it — answering a spoken request
        from the typed prompt alone, with no error anywhere."""
        from tui_gateway import compute_host

        class StaleWorker:
            def _run_prompt_submit(self, rid, sid, session, text):  # pre-PAN-49 signature
                return None

        assert compute_host._worker_carries_voice_live(StaleWorker()) is False

    def test_the_frame_carries_the_per_turn_surface_including_the_app_window(self):
        session = _session(running=False)

        assert server._compute_host_turn_frame(
            "r1", "sid", session, "text", client_surface="hud"
        )["client_surface"] == "hud"
        # "" is a real value (the app window), not an absent field.
        assert server._compute_host_turn_frame(
            "r1", "sid", session, "text", client_surface=""
        )["client_surface"] == ""
        assert "client_surface" not in server._compute_host_turn_frame(
            "r1", "sid", session, "text"
        )

    def test_a_worker_without_the_surface_parameter_fails_closed(self):
        from tui_gateway import compute_host

        class StaleWorker:
            def _run_prompt_submit(self, rid, sid, session, text, voice_live_turn=None):
                return None

        assert compute_host._worker_carries_client_surface(StaleWorker()) is False
        assert compute_host._worker_carries_client_surface(server) is True


class TestSurfaceDoesNotBleedBetweenQueuedTurns:
    """R2: which window a turn was typed into is a fact about that turn.

    ``client_surface`` is still mirrored on the session, but a queued turn drains after the
    NEXT submit has already rewritten that mirror. Without carrying the surface on the
    envelope, the drained turn renders whichever window came last — so a HUD submit queued
    behind a voice delegation (or the reverse) silently annotates the wrong turn.
    """

    @staticmethod
    def _drain(monkeypatch, session):
        """Drain the queue, capturing what each turn was actually handed."""
        captured = []
        monkeypatch.setattr(server, "_session_uses_compute_host", lambda *a, **k: False)
        monkeypatch.setattr(
            server,
            "_run_prompt_submit",
            lambda rid, sid, sess, text, **kwargs: captured.append((text, kwargs)) or True,
        )
        while True:
            session["running"] = False
            if not server._drain_queued_prompt("r-drain", "sid", session):
                break
        return captured

    def test_voice_then_hud_keeps_both_surfaces(self, busy_session, monkeypatch):
        _submit(surface="voice-live", voice_context="User: spoken", text="spoken")
        _submit(surface="hud", text="typed in the HUD")

        assert busy_session["queued_prompt"]["client_surface"] == "voice-live"
        assert busy_session["queued_prompts"][0]["client_surface"] == "hud"

        captured = self._drain(monkeypatch, busy_session)

        assert [(text, kw["client_surface"]) for text, kw in captured] == [
            ("spoken", "voice-live"),
            ("typed in the HUD", "hud"),
        ]
        assert captured[0][1]["voice_live_turn"]["voice_context"] == "User: spoken"
        assert captured[1][1].get("voice_live_turn") is None

    def test_hud_then_voice_keeps_both_surfaces(self, busy_session, monkeypatch):
        _submit(surface="hud", text="typed in the HUD")
        _submit(surface="voice-live", voice_context="User: spoken", text="spoken")

        assert busy_session["queued_prompt"]["client_surface"] == "hud"
        assert busy_session["queued_prompts"][0]["client_surface"] == "voice-live"

        captured = self._drain(monkeypatch, busy_session)

        assert [(text, kw["client_surface"]) for text, kw in captured] == [
            ("typed in the HUD", "hud"),
            ("spoken", "voice-live"),
        ]

    def test_a_turn_source_with_no_surface_still_falls_back_to_the_mirror(self, busy_session):
        """The per-turn value is an override, not a replacement: sources that pin nothing
        (auto-continue, re-anchor) keep reading the session marker."""
        busy_session["client_surface"] = "hud"

        _submit(text="from the HUD", surface="hud")
        assert busy_session["queued_prompt"]["client_surface"] == "hud"

    def test_the_per_turn_surface_overrides_a_stale_session_mirror(self):
        """The bleed itself: the mirror says hud (a later submit), the turn says app window."""
        session = {
            "client_surface": "hud",
            # The HUD note is withheld without the tool it rests on, so the fixture
            # agent carries one — otherwise every assertion below is "" for the
            # wrong reason.
            "agent": types.SimpleNamespace(
                valid_tool_names={"read_window_below", "computer_use"}
            ),
        }

        assert server._hud_surface_note(session, "hud")
        assert server._hud_surface_note(session, "") == ""
        assert server._hud_surface_note(session, "voice-live") == ""
        # Nothing pinned -> the pre-existing mirror seam still applies.
        assert server._hud_surface_note(session)

    def test_two_surfaces_are_never_merged_into_one_queued_turn(self, busy_session):
        """Merging across surfaces would either graft the HUD note onto app-window words or
        drop it from the HUD words it belongs to."""
        _submit(surface="hud", text="first")
        _submit(surface="", text="second")

        assert busy_session["queued_prompt"]["text"] == "first"
        assert busy_session["queued_prompts"][0]["text"] == "second"
        assert busy_session["queued_prompts"][0]["client_surface"] == ""


class TestVoiceLiveSkipsServerTts:
    """GPT-Live delegations are spoken by the Live client — server TTS would double-speak."""

    def test_voice_live_turn_skips_begin_turn_tts(self, monkeypatch):
        begin_calls = []
        tts_puts = []

        def fake_begin(sid):
            begin_calls.append(sid)
            mock_q = types.SimpleNamespace(
                put=lambda delta: tts_puts.append(delta),
            )
            return mock_q, None

        monkeypatch.setattr(server, "_begin_turn_tts", fake_begin)
        monkeypatch.setattr(server, "_ensure_active_session_slot", lambda *a, **k: None)
        monkeypatch.setattr(server, "_emit", lambda *a, **k: None)
        monkeypatch.setattr(
            server,
            "_start_usage_ticker",
            lambda *a, **k: (threading.Event(), types.SimpleNamespace(join=lambda: None)),
        )
        monkeypatch.setattr(server, "_set_session_context", lambda *a, **k: [])
        monkeypatch.setattr(server, "_wire_callbacks", lambda *a, **k: None)
        monkeypatch.setattr(server, "_apply_pending_model_switch", lambda *a, **k: None)
        monkeypatch.setattr(server, "_sync_agent_model_with_config", lambda *a, **k: None)
        monkeypatch.setattr(server, "_sync_agent_compression_with_config", lambda *a, **k: None)
        monkeypatch.setattr(server, "_sync_agent_provider_credentials_with_config", lambda *a, **k: None)
        monkeypatch.setattr(server, "_sync_bot_capabilities", lambda *a, **k: None)
        monkeypatch.setattr(server, "_session_cwd", lambda *a, **k: "/tmp")
        monkeypatch.setattr(server, "_register_session_cwd", lambda *a, **k: None)
        monkeypatch.setattr(server, "make_stream_renderer", lambda *a, **k: None)
        monkeypatch.setattr(server, "_voice_mode_enabled", lambda: True)
        monkeypatch.setattr(server, "_voice_cfg_dict", lambda: {"barge_in": False})
        monkeypatch.setattr(server, "_load_interim_assistant_messages", lambda: False)
        monkeypatch.setattr(server, "_hud_surface_note", lambda *a, **k: "")
        monkeypatch.setattr(server, "_pending_reaction_notes", lambda *a: "")
        monkeypatch.setattr(server, "_prepend_note", lambda msg, note: msg if not note else f"{note}\n{msg}")
        monkeypatch.setattr(server, "_resolve_model", lambda: "test-model")
        monkeypatch.setattr(server, "record_turn_start", lambda *a, **k: None)
        monkeypatch.setattr(server, "clear_turn_marker", lambda *a, **k: None)
        monkeypatch.setattr(server, "_session_home", lambda *a: types.SimpleNamespace())
        monkeypatch.setattr(server, "_retire_turn_marker", lambda *a, **k: None)
        monkeypatch.setattr(server, "_set_session_not_running", lambda *a, **k: None)

        class _SyncThread:
            def __init__(self, target=None, daemon=None):
                self._target = target

            def start(self):
                if self._target:
                    self._target()

            def is_alive(self):
                return False

        monkeypatch.setattr(server.threading, "Thread", _SyncThread)

        def run_conv(msg, **kwargs):
            stream_cb = kwargs.get("stream_callback")
            if stream_cb:
                stream_cb("spoken chunk")
            return {"messages": []}

        agent = types.SimpleNamespace(
            clear_interrupt=lambda: None,
            run_conversation=run_conv,
            session_id="agent-sid",
            valid_tool_names=set(),
            model="test",
            base_url="",
            api_key="",
            provider="",
        )
        session = _session(agent=agent, running=True, session_key="sk")
        server._sessions["sid-vlive"] = session

        voice_live_turn = {"surface": "voice-live", "voice_context": "User: hi"}
        try:
            server._run_prompt_submit(
                "rid",
                "sid-vlive",
                session,
                "delegate text",
                voice_live_turn=voice_live_turn,
            )
            assert begin_calls == []
            assert tts_puts == []
        finally:
            server._sessions.pop("sid-vlive", None)

    def test_plain_turn_still_begins_server_tts(self, monkeypatch):
        begin_calls = []

        monkeypatch.setattr(
            server,
            "_begin_turn_tts",
            lambda sid: begin_calls.append(sid) or (None, None),
        )
        monkeypatch.setattr(server, "_ensure_active_session_slot", lambda *a, **k: None)
        monkeypatch.setattr(server, "_emit", lambda *a, **k: None)
        monkeypatch.setattr(
            server,
            "_start_usage_ticker",
            lambda *a, **k: (threading.Event(), types.SimpleNamespace(join=lambda: None)),
        )
        monkeypatch.setattr(server, "_set_session_context", lambda *a, **k: [])
        monkeypatch.setattr(server, "_wire_callbacks", lambda *a, **k: None)
        monkeypatch.setattr(server, "_apply_pending_model_switch", lambda *a, **k: None)
        monkeypatch.setattr(server, "_sync_agent_model_with_config", lambda *a, **k: None)
        monkeypatch.setattr(server, "_sync_agent_compression_with_config", lambda *a, **k: None)
        monkeypatch.setattr(server, "_sync_agent_provider_credentials_with_config", lambda *a, **k: None)
        monkeypatch.setattr(server, "_sync_bot_capabilities", lambda *a, **k: None)
        monkeypatch.setattr(server, "_session_cwd", lambda *a, **k: "/tmp")
        monkeypatch.setattr(server, "_register_session_cwd", lambda *a, **k: None)
        monkeypatch.setattr(server, "make_stream_renderer", lambda *a, **k: None)
        monkeypatch.setattr(server, "_voice_mode_enabled", lambda: True)
        monkeypatch.setattr(server, "_voice_cfg_dict", lambda: {"barge_in": False})
        monkeypatch.setattr(server, "_load_interim_assistant_messages", lambda: False)
        monkeypatch.setattr(server, "_hud_surface_note", lambda *a, **k: "")
        monkeypatch.setattr(server, "_pending_reaction_notes", lambda *a: "")
        monkeypatch.setattr(server, "_prepend_note", lambda msg, note: msg)
        monkeypatch.setattr(server, "_resolve_model", lambda: "test-model")
        monkeypatch.setattr(server, "record_turn_start", lambda *a, **k: None)
        monkeypatch.setattr(server, "clear_turn_marker", lambda *a, **k: None)
        monkeypatch.setattr(server, "_session_home", lambda *a: types.SimpleNamespace())
        monkeypatch.setattr(server, "_retire_turn_marker", lambda *a, **k: None)
        monkeypatch.setattr(server, "_set_session_not_running", lambda *a, **k: None)

        class _SyncThread:
            def __init__(self, target=None, daemon=None):
                self._target = target

            def start(self):
                if self._target:
                    self._target()

            def is_alive(self):
                return False

        monkeypatch.setattr(server.threading, "Thread", _SyncThread)

        agent = types.SimpleNamespace(
            clear_interrupt=lambda: None,
            run_conversation=lambda *a, **k: {"messages": []},
            session_id="agent-sid",
            valid_tool_names=set(),
            model="test",
            base_url="",
            api_key="",
            provider="",
        )
        session = _session(agent=agent, running=True, session_key="sk")
        server._sessions["sid-plain"] = session

        try:
            server._run_prompt_submit("rid", "sid-plain", session, "hello")
            assert begin_calls == ["sid-plain"]
        finally:
            server._sessions.pop("sid-plain", None)


class TestNonStringSurface:
    """R5: ``["hud"] not in SURFACES`` is a TypeError, not a tolerated unknown surface."""

    @pytest.mark.parametrize("raw", [["hud"], {"surface": "hud"}, 1, 1.5, True, b"hud"])
    def test_a_non_string_surface_is_refused_as_invalid_params(self, busy_session, raw):
        response = _submit(surface=raw)

        assert response["error"]["code"] == 4004
        assert "surface" in response["error"]["message"]
        assert not busy_session.get("queued_prompt")

    def test_an_unknown_surface_name_is_still_tolerated(self, busy_session):
        """Deliberate fork divergence: every non-HUD client simply omits the field, so an
        unrecognised NAME is read as "no surface" rather than refused."""
        response = _submit(surface="pet-overlay")

        assert "error" not in response
        assert busy_session["client_surface"] == ""

    def test_the_shared_admission_helper_refuses_an_unhashable_surface(self):
        """Defence in depth: the RPC path is not the only caller of ``admit_surface``."""
        from gateway.session_surface import SurfaceAdmissionError, admit_surface

        for raw in (["hud"], {"surface": "hud"}, 7):
            with pytest.raises(SurfaceAdmissionError):
                admit_surface({"surface": raw})