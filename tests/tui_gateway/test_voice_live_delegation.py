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

import json
import threading
import types

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
                        "transport": {"type": "webrtc", "sdp": "answer"},
                    }
                ).encode()
            )

        monkeypatch.setattr(voice_live.urllib.request, "urlopen", fake_urlopen)
        monkeypatch.setattr(
            voice_live,
            "_live_section",
            lambda voice=None: {"voice": "willow", "instructions": "Speak Spanish."},
        )
        monkeypatch.setattr(
            voice_live, "_resolve_credentials", lambda live: ("sk-test", "https://api.example/v1")
        )

        result = voice_live.create_webrtc_session(
            "v=0 offer", history=[{"type": "message", "role": "user", "content": []}]
        )

        assert result["transport"]["sdp"] == "answer"
        assert captured["url"] == "https://api.example/v1/live/sessions"
        assert captured["auth"] == "Bearer sk-test"
        session = captured["body"]["session"]
        assert session["delegation"] == {"type": "client"}
        assert session["audio"]["output"]["voice"] == "willow"
        assert session["instructions"].endswith("Speak Spanish.")
        assert session["input"][0]["role"] == "user"
        assert captured["body"]["transport"] == {"type": "webrtc", "sdp": "v=0 offer"}
        assert "sk-test" not in json.dumps(result)

    def test_missing_key_refuses_before_any_network(self, monkeypatch):
        monkeypatch.setattr(
            voice_live,
            "_resolve_credentials",
            lambda live: ("", voice_live.DEFAULT_LIVE_BASE_URL),
        )
        monkeypatch.setattr(
            voice_live.urllib.request, "urlopen", lambda *a, **k: pytest.fail("must not call the vendor")
        )

        with pytest.raises(ValueError):
            voice_live.create_webrtc_session("v=0 offer")
        assert voice_live.resolve_gpt_live_status()["available"] is False

    def test_sdp_offer_is_forwarded_byte_exact(self, monkeypatch):
        """A stripped offer answers 400 "failed to unmarshal SDP: EOF" at the vendor — the
        offer goes over the wire exactly as produced, trailing CRLF included."""
        captured = {}

        class _Resp:
            def read(self):
                return b"{}"

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        def fake_urlopen(req, timeout=0):
            captured["sdp"] = json.loads(req.data)["transport"]["sdp"]
            return _Resp()

        monkeypatch.setattr(voice_live.urllib.request, "urlopen", fake_urlopen)
        monkeypatch.setattr(voice_live, "_resolve_credentials", lambda live: ("sk-test", "https://x/v1"))
        offer = "v=0\r\no=- 1 1 IN IP4 127.0.0.1\r\ns=-\r\n"

        voice_live.create_webrtc_session(offer)

        assert captured["sdp"] == offer


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