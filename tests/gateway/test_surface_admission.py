"""Surface admission: the spoken transcript is admitted with one turn, or refused.

A ``surface`` is a fact about ONE turn, never about the session. The dangerous
field is ``voice_context``: it is model input — the words the model reads beside
the typed prompt — so a client able to attach it to an arbitrary surface could
put text into a turn the user never dictated, with nothing in the transcript to
show for it. Admission accepts it on ``voice-live`` and refuses it everywhere
else.

Fork port notes: ``restore_surface()`` is not ported (it re-validates a committed
surface crossing a managed-worker process boundary, which needs the durable
``hermes_state_runtime`` store this fork does not have), and
``RuntimeStoreError`` is the fork-local ``SurfaceAdmissionError``.
"""

import pytest

from gateway.session_surface import (
    SURFACES,
    VOICE_CONTEXT_LIMIT,
    SurfaceAdmissionError,
    admit_surface,
    submit_surface_fields,
    surface_note,
)
from tools.voice_live import VOICE_LIVE_TURN_NOTE

TOOLKIT = {"read_window_below", "computer_use"}


class TestAdmission:
    def test_a_plain_window_submit_commits_nothing(self):
        assert admit_surface({}) == {}

    def test_every_known_surface_is_admitted(self):
        assert SURFACES == {"hud", "voice-live"}

        for name in SURFACES:
            assert admit_surface({"surface": name}) == {"surface": name}

    def test_an_unknown_surface_is_refused(self):
        with pytest.raises(SurfaceAdmissionError):
            admit_surface({"surface": "pet-overlay"})

    def test_a_live_turn_commits_its_transcript(self):
        committed = admit_surface({"surface": "voice-live", "voice_context": "User: hi"})

        assert committed == {"surface": "voice-live", "voice_context": "User: hi"}

    def test_an_oversized_transcript_is_clipped_to_the_limit(self):
        committed = admit_surface(
            {"surface": "voice-live", "voice_context": "x" * (VOICE_CONTEXT_LIMIT + 500)}
        )

        assert len(committed["voice_context"]) == VOICE_CONTEXT_LIMIT

    def test_boolean_flags_must_be_booleans(self):
        with pytest.raises(SurfaceAdmissionError):
            admit_surface({"surface": "voice-live", "voice_turn": "yes"})
        with pytest.raises(SurfaceAdmissionError):
            admit_surface({"surface": "voice-live", "interrupted": 1})


class TestTranscriptSmugglingGuard:
    """The control the fork refuses to relax."""

    @pytest.mark.parametrize("surface", [None, "", "hud", "pet-overlay"])
    def test_a_transcript_off_the_live_surface_is_refused(self, surface):
        params = {"voice_context": "User: smuggled"}
        if surface is not None:
            params["surface"] = surface

        with pytest.raises(SurfaceAdmissionError):
            admit_surface(params)

    def test_a_non_string_transcript_is_refused(self):
        with pytest.raises(SurfaceAdmissionError):
            admit_surface({"surface": "voice-live", "voice_context": {"role": "user"}})

    def test_an_empty_transcript_on_the_wrong_surface_is_still_refused(self):
        """Refusing is about the field's presence, not its content: an empty string is
        still a client reaching for a channel it does not own."""
        with pytest.raises(SurfaceAdmissionError):
            admit_surface({"surface": "hud", "voice_context": ""})

    def test_an_absent_transcript_is_fine_anywhere(self):
        assert admit_surface({"surface": "hud"}) == {"surface": "hud"}


class TestNoteIsolation:
    """One surface's note must never reach another surface's turn."""

    def test_the_live_turn_gets_the_spoken_delegation_contract(self):
        note = surface_note({"surface": "voice-live", "voice_context": "User: hi"})

        assert note.startswith(VOICE_LIVE_TURN_NOTE)
        assert "User: hi" in note

    def test_the_hud_turn_never_gets_the_spoken_contract(self):
        note = surface_note({"surface": "hud"}, TOOLKIT)

        assert "delegation" not in note
        assert VOICE_LIVE_TURN_NOTE not in note
        assert "read_window_below" in note

    def test_the_app_window_gets_nothing(self):
        assert surface_note({}) == ""
        assert surface_note(None) == ""

    def test_barge_in_precedes_the_surface_contract(self):
        from tools.tts_streaming import SPEECH_INTERRUPTED_NOTE

        note = surface_note(
            {"surface": "voice-live", "interrupted": True, "voice_context": "User: hi"}
        )

        assert note.startswith(SPEECH_INTERRUPTED_NOTE)
        assert VOICE_LIVE_TURN_NOTE in note


class TestSubmitSurfaceFields:
    def test_only_the_per_turn_fields_are_forwarded(self):
        params = {
            "text": "hello",
            "session_id": "sid",
            "surface": "voice-live",
            "voice_context": "User: hi",
            "voice_turn": True,
        }

        forwarded = submit_surface_fields(params)

        assert forwarded == {
            "surface": "voice-live",
            "voice_context": "User: hi",
            "voice_turn": True,
        }

    def test_a_submit_with_none_of_them_forwards_nothing(self):
        assert submit_surface_fields({"text": "hello"}) == {}