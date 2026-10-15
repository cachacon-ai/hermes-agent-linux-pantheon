"""Per-turn client surface (floating HUD, live voice) and its model-bound note.

A surface is a fact about ONE turn, never about the session: the same session
alternates between the app window, the HUD and a spoken delegation. It is
admitted once, at ``prompt.submit``, and carried as per-turn parameters, so the
persisted user row stays the words the user typed and the frozen system prefix
is untouched.

Fork port notes (from upstream ``gateway/session_surface.py``):

* ``restore_surface()`` is dropped — upstream uses it to re-validate a committed
  ``surface_v1`` arriving across a process boundary from a managed worker, which
  depends on the durable ``hermes_state_runtime`` store. That module does not
  exist in this fork, so there is no cross-process contract to re-validate and
  the function would be dead code.
* ``RuntimeStoreError`` is replaced by the fork-local ``SurfaceAdmissionError``.
* The ``ContextVar`` turn-scope helpers (``surface_turn_scope``,
  ``surface_voice_turn``, ``surface_turn_note``, ``arm_surface_turn``) are dropped:
  this fork threads per-turn surface data explicitly through the turn call, so a
  contextvar side-channel would be an unused second path. The HUD note keeps its
  existing ``_hud_surface_note`` seam in ``tui_gateway/server.py``; only the
  voice-live turn passes through ``surface_note()`` here, so the HUD note is
  never emitted twice.
"""
from __future__ import annotations

SURFACES = frozenset({"hud", "voice-live"})
VOICE_CONTEXT_LIMIT = 6000
_SUBMIT_FIELDS = ("surface", "voice_context", "interrupted", "voice_turn")


class SurfaceAdmissionError(ValueError):
    """A ``prompt.submit`` carried surface fields admission cannot accept.

    Subclasses ``ValueError`` so the RPC handlers can catch it without importing
    a dedicated type, and so it can never be confused with the plain-text
    argument errors of the surrounding parse helpers.
    """


def submit_surface_fields(params):
    """The raw per-turn surface fields of a ``prompt.submit`` request."""
    return {key: params[key] for key in _SUBMIT_FIELDS if key in params}


def admit_surface(params):
    """Wire ``surface`` / ``voice_context`` / ``interrupted`` / ``voice_turn`` into a committed
    per-turn surface dict (``{}`` when the submit carries none). ``voice_turn`` marks a spoken
    voice-mode turn.

    Raises ``SurfaceAdmissionError`` when a field cannot be admitted — in
    particular when ``voice_context`` arrives on anything other than a
    ``voice-live`` surface, which is a client smuggling model input past the
    persisted row rather than a transcript.
    """
    surface, context, interrupted, voice_turn = (params.get(key) for key in _SUBMIT_FIELDS)
    if surface is not None and surface not in SURFACES:
        raise SurfaceAdmissionError("invalid_params")
    # The spoken transcript only makes sense for a live-voice delegation; anywhere
    # else it is a client smuggling model input past the persisted row.
    if context is not None and (surface != "voice-live" or not isinstance(context, str)):
        raise SurfaceAdmissionError("invalid_params")
    if any(flag is not None and type(flag) is not bool for flag in (interrupted, voice_turn)):
        raise SurfaceAdmissionError("invalid_params")
    committed: dict = {}
    if surface:
        committed["surface"] = surface
    if context:
        committed["voice_context"] = context[:VOICE_CONTEXT_LIMIT]
    if interrupted:
        committed["interrupted"] = True
    if voice_turn:
        committed["voice_turn"] = True
    return committed


def _hud_note(committed, valid_tool_names):
    from agent.prompt_builder import hud_surface_note

    return hud_surface_note(valid_tool_names)


def _voice_live_note(committed, valid_tool_names):
    from tools.voice_live import voice_live_turn_note

    return voice_live_turn_note(committed.get("voice_context") or "")


_SURFACE_NOTES = {"hud": _hud_note, "voice-live": _voice_live_note}


def surface_note(committed, valid_tool_names=None):
    """The model-bound note for an admitted surface (``""`` for the plain app window): barge-in
    first, then the HUD read-the-window-below prior or the spoken-delegation contract."""
    committed = committed or {}
    notes = []
    if committed.get("interrupted"):
        from tools.tts_streaming import SPEECH_INTERRUPTED_NOTE

        notes.append(SPEECH_INTERRUPTED_NOTE)
    render = _SURFACE_NOTES.get(committed.get("surface"))
    if render is not None:
        notes.append(render(committed, valid_tool_names))
    return "\n\n".join(note for note in notes if note)