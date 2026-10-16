"""GPT-Live voice chat mode: the full-duplex voice frontend that delegates to Hermes.

``voice.voice_chat_mode: gpt-live`` replaces the chained STT → turn → TTS loop with ONE
full-duplex voice model (OpenAI ``gpt-live-1``) that owns the microphone and the speaker and
delegates every real request to Hermes as its *client-delegation* backend. Hermes stays the
agent: whatever model/provider the session has selected answers, with the full toolset.

Division of labour (the Live API has no tools of its own in client mode):

* the desktop renderer holds the WebRTC media session (mic in, speech out) and the data channel;
* this module resolves WHICH credentials/voice/persona to use and performs the one server-side
  step the API requires — exchanging the browser's SDP offer for an answer with the project key
  (``POST /v1/live/sessions``), so the key never reaches the client;
* the renderer turns each ``session.delegation.created`` into a normal ``prompt.submit`` on the
  active session (surface ``voice-live``) and streams the reply back as
  ``session.commentary.append`` — Hermes' answer is what the voice speaks.

Vendor contract: https://developers.openai.com/api/docs/guides/live (+ live-delegation,
voice-webrtc). Billing is $0.05/min of session time on the OpenAI key, separate from the
Hermes turn.
"""

from __future__ import annotations

import json
import logging
import re
import urllib.error
import urllib.request
from typing import Any, Dict, Optional
from urllib.parse import quote

logger = logging.getLogger(__name__)

GPT_LIVE_MODE = "gpt-live"
CHAINED_MODE = "chained"

# Vendor limits for the ``input`` history of a live session (live-conversations guide):
# at most 128 messages, ~8192 combined tokens. Sizes are enforced by pruning the OLDEST
# messages; a structurally invalid entry is a client error and is refused instead.
LIVE_HISTORY_MAX_MESSAGES = 128
LIVE_HISTORY_MAX_TOKENS = 8192
# Conservative token estimate: 4 characters per token over-estimates rather than
# under-estimates, so pruning happens before the vendor rejects the session.
LIVE_HISTORY_CHARS_PER_TOKEN = 4
LIVE_HISTORY_ROLES = frozenset({"system", "user", "assistant"})
# The only content parts a live session accepts. Anything else (audio, image, tool
# output) is not text the voice model can be primed with.
LIVE_HISTORY_PART_TYPES = frozenset({"text", "input_text"})

DEFAULT_LIVE_MODEL = "gpt-live-1"
DEFAULT_LIVE_VOICE = "marin"
DEFAULT_LIVE_BASE_URL = "https://api.openai.com/v1"
# Vendor session ids are opaque tokens (e.g. ``live_u2_…``). Restrict to one URL path
# segment so an authenticated hangup cannot be redirected by ``/``, ``?``, ``#``, or
# encoded separators in the request body.
LIVE_VENDOR_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
# Voices the vendor lists for gpt-live-1 (live-conversations guide) plus the realtime defaults it
# accepts; free text stays allowed for custom voices.
GPT_LIVE_VOICES = (
    "marin", "cedar", "quartz", "ripple", "vesper", "willow", "stone", "gleam", "meridian",
    "bossa", "tempo", "beacon", "delta", "cinder",
)

# Persona for the voice layer. Short on purpose: the live model has a small context window and
# the vendor guide asks for role + style + a labelled delegation policy, nothing more. The
# backend (Hermes) carries the real instructions, tools and memory.
LIVE_PERSONA = (
    "You are Hermes, a calm and friendly voice assistant. Speak naturally at an unhurried pace. "
    "Be clear and direct, not overly cheerful. If the user is frustrated, acknowledge it briefly "
    "and focus on the next helpful step.\n\n"
    "Backchannel policy: Use moderate backchannels. Acknowledge naturally without competing with "
    "the main response.\n\n"
    "Interruption policy: Stop speaking when the user interrupts. Listen to what they say.\n\n"
    "Delegation policy:\n"
    "Backend tools:\n"
    "- Hermes agent: a full AI agent with tools — it can run commands, read and edit files, "
    "browse the web, search, remember things across sessions, schedule tasks, and reason "
    "carefully about anything. It is the one who actually does work and knows facts.\n\n"
    "Delegate to the backend when:\n"
    "- The user asks a question that needs facts, current information, or careful reasoning.\n"
    "- The user asks you to do, check, find, make, fix, run or remember anything.\n"
    "- A correction changes work already requested.\n\n"
    "Do not delegate to the backend when:\n"
    "- The user greets you, makes small talk, or asks you to repeat a result already provided.\n"
    "- You need a brief clarification to understand the request.\n\n"
    "Delegate before giving an answer that depends on backend work. Do not guess the result "
    "while waiting; say briefly that you are checking, then wait for the result."
)

# Per-turn note prepended to the MODEL INPUT (never the byte-stable system prompt) when a turn
# arrives from the live voice layer. Same seam as the HUD note.
VOICE_LIVE_TURN_NOTE = (
    "[Note: this message is a delegation from a live spoken conversation. The text is a voice "
    "transcript (it may contain mis-hearings, hesitations and later corrections; use the latest "
    "intent). Your reply will be spoken aloud by a voice model that paraphrases it: answer in plain "
    "conversational sentences, keep it short (a few sentences unless the user asked for detail), no "
    "markdown, no lists, no code blocks, no URLs read out character by character. Do the work with "
    "your tools as usual; only the final facts need to be spoken. Do not claim an action succeeded "
    "before it actually did.]"
)


def voice_live_turn_note(context: str = "") -> str:
    """The per-turn note plus, when the client sent one, the recent spoken exchange the delegation
    refers to (the user's last words alone are often "yes" or "Thursday, not Friday")."""
    context = context.strip()
    if not context:
        return VOICE_LIVE_TURN_NOTE
    return f"{VOICE_LIVE_TURN_NOTE}\n[Recent spoken conversation, newest last:\n{context}]"


def _voice_section() -> dict[str, Any]:
    try:
        from hermes_cli.config import load_config
        voice = load_config().get("voice")
    except Exception:
        return {}
    return voice if isinstance(voice, dict) else {}


def _live_section(voice: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    section = (voice if voice is not None else _voice_section()).get("gpt_live")
    return section if isinstance(section, dict) else {}


def voice_chat_mode(voice: Optional[dict[str, Any]] = None) -> str:
    """``chained`` (default) or ``gpt-live``. Accepts the underscore spelling too."""
    raw = (voice if voice is not None else _voice_section()).get("voice_chat_mode")
    mode = str(raw or CHAINED_MODE).strip().lower().replace("_", "-")
    return GPT_LIVE_MODE if mode in {GPT_LIVE_MODE, "gptlive", "live"} else CHAINED_MODE


class VoiceLiveError(RuntimeError):
    """A typed GPT-Live failure. The HTTP route maps each subclass to its own status
    and a fixed detail string, so a vendor body never reaches the client."""


class VoiceLiveDisabledError(VoiceLiveError):
    """``voice.voice_chat_mode`` is not ``gpt-live``. Raised BEFORE any vendor call: this
    mode is opt-in, and a chained install must never start (or bill) a live session."""


class VoiceLiveProtocolError(VoiceLiveError):
    """The vendor answered 2xx with a body that is not a usable session answer (no session
    id, or no SDP answer). A failed session, not a missing key."""


class VoiceLiveUpstreamError(VoiceLiveError):
    """The vendor rejected the request or the transport failed. Carries the HTTP status and
    an error category only — never the upstream body, which is untrusted and may echo key
    material."""


class VoiceLiveHistoryError(ValueError):
    """A ``history`` entry is structurally unusable (not a message, a disallowed role, or
    non-text content). Subclasses ``ValueError`` so it cannot be mistaken for the missing-key
    case, and is raised as a client error (400) rather than pruned."""


class VoiceLiveMissingKeyError(ValueError):
    """No OpenAI key resolves for this profile. The client falls back to chained mode."""


def ensure_gpt_live_mode(voice: Optional[dict[str, Any]] = None) -> str:
    """Refuse unless the profile actually selected ``gpt-live``.

    The mode gate lives here, ahead of credential resolution and the network call, so a
    chained install cannot start a live session even when a key happens to resolve. The
    opt-in is the contract: a key alone does not enable this mode.
    """
    mode = voice_chat_mode(voice)
    if mode != GPT_LIVE_MODE:
        raise VoiceLiveDisabledError(
            f"voice chat mode is {CHAINED_MODE!r}; set voice.voice_chat_mode: {GPT_LIVE_MODE} "
            "to use GPT-Live"
        )
    return mode


def _resolve_credentials(live: dict[str, Any]) -> tuple[str, str]:
    """``(api_key, base_url)`` — ``voice.gpt_live.api_key`` first, else the same OpenAI audio
    chain the STT/TTS providers use (``VOICE_TOOLS_OPENAI_KEY`` → ``OPENAI_API_KEY`` → pool).

    The Nous-managed audio proxy does not carry ``/live/sessions``; this mode is direct-key only.
    """
    from tools.tool_backend_helpers import resolve_openai_audio_api_key
    api_key = str(live.get("api_key") or "").strip() or resolve_openai_audio_api_key()
    base_url = str(live.get("base_url") or DEFAULT_LIVE_BASE_URL).strip().rstrip("/")
    return api_key, base_url


def live_instructions(live: Optional[dict[str, Any]] = None) -> str:
    extra = str((live if live is not None else _live_section()).get("instructions") or "").strip()
    return f"{LIVE_PERSONA}\n\n{extra}" if extra else LIVE_PERSONA


def resolve_gpt_live_status() -> dict[str, Any]:
    """Non-secret readiness verdict for the client: which mode is selected and whether GPT-Live
    can start (a key resolves). Never returns the key."""
    voice = _voice_section()
    mode = voice_chat_mode(voice)
    live = _live_section(voice)
    api_key, _base = _resolve_credentials(live)
    return {
        "mode": mode,
        "available": bool(api_key),
        "reason": None if api_key else "no OpenAI API key (set OPENAI_API_KEY or voice.gpt_live.api_key)",
        "model": str(live.get("model") or DEFAULT_LIVE_MODEL),
        "voice": str(live.get("voice") or DEFAULT_LIVE_VOICE),
    }


def _history_message_text(message: dict[str, Any], index: int) -> str:
    """The single text body of a history message.

    The live API accepts ONE text part per message and text content only. A bare string
    content is the common shape; a parts list is accepted only when every part is text,
    and is joined so the message still carries one text part.
    """
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts: list[str] = []
        for part in content:
            if not isinstance(part, dict):
                raise VoiceLiveHistoryError(f"history[{index}] has a non-object content part")
            part_type = part.get("type")
            text = part.get("text")
            if part_type not in LIVE_HISTORY_PART_TYPES or not isinstance(text, str):
                raise VoiceLiveHistoryError(
                    f"history[{index}] content must be text (type and text string)"
                )
            texts.append(text)
        return "\n".join(texts)
    raise VoiceLiveHistoryError(f"history[{index}] content must be a string or a list of text parts")


def normalize_session_history(history: Optional[list]) -> list[dict[str, Any]]:
    """Validate and prune the client-supplied ``input`` history for a live session.

    Structural violations are REFUSED (400), not silently dropped: a message whose role or
    content the voice model cannot be primed with is a client bug, and quietly discarding it
    would leave the model primed on a history the user never saw. SIZE limits are pruned
    instead — the oldest messages go first, which is what the vendor itself asks for.
    """
    if not history:
        return []
    if not isinstance(history, list):
        raise VoiceLiveHistoryError("history must be a list of messages")
    normalized: list[dict[str, Any]] = []
    for index, message in enumerate(history):
        if not isinstance(message, dict):
            raise VoiceLiveHistoryError(f"history[{index}] must be an object")
        role = message.get("role")
        if not isinstance(role, str) or role.strip().lower() not in LIVE_HISTORY_ROLES:
            raise VoiceLiveHistoryError(
                f"history[{index}] role must be one of {sorted(LIVE_HISTORY_ROLES)}"
            )
        text = _history_message_text(message, index)
        if not text.strip():
            # An empty message has nothing to prime with and costs a slot in a 128-message
            # budget; drop it rather than pruning a real one later.
            continue
        normalized.append(
            {
                "type": "message",
                "role": role.strip().lower(),
                "content": [{"type": "input_text", "text": text}],
            }
        )
    # Prune oldest-first against the vendor's count and token budget. The token estimate
    # is deliberately coarse (4 chars/token) so it prunes early rather than late.
    pruned = normalized[-LIVE_HISTORY_MAX_MESSAGES:]
    budget_chars = LIVE_HISTORY_MAX_TOKENS * LIVE_HISTORY_CHARS_PER_TOKEN
    total_chars = sum(
        len(part["text"]) for message in pruned for part in message["content"]
    )
    while len(pruned) > 1 and total_chars > budget_chars:
        dropped = pruned.pop(0)
        total_chars -= len(dropped["content"][0]["text"])
    return pruned


def build_session_config(history: Optional[list] = None) -> dict[str, Any]:
    """The ``session`` object for ``POST /v1/live/sessions`` (client delegation, WebRTC — the
    transport negotiates the audio format, so none is set)."""
    live = _live_section()
    config: dict[str, Any] = {
        "model": str(live.get("model") or DEFAULT_LIVE_MODEL),
        "instructions": live_instructions(live),
        "audio": {"output": {"voice": str(live.get("voice") or DEFAULT_LIVE_VOICE)}},
        "delegation": {"type": "client"},
    }
    normalized = normalize_session_history(history)
    if normalized:
        config["input"] = normalized
    return config


def _validate_session_answer(payload: Any) -> dict[str, Any]:
    """Require a usable session answer: a non-empty session id and a WebRTC SDP answer.

    A 200 with an unexpected body is a FAILED session reported as a success — the client
    would mount a media session against a payload that has no answer. Both fields are
    checked for shape here so the failure is one typed error (mapped to 502) instead of a
    confusing WebRTC error much later.
    """
    if not isinstance(payload, dict):
        raise VoiceLiveProtocolError("vendor answer is not an object")
    session = payload.get("session")
    session_id = session.get("id") if isinstance(session, dict) else None
    if not isinstance(session_id, str) or not session_id.strip():
        raise VoiceLiveProtocolError("vendor answer has no session id")
    transport = payload.get("transport")
    if not isinstance(transport, dict):
        raise VoiceLiveProtocolError("vendor answer has no transport")
    sdp = transport.get("sdp")
    if not isinstance(sdp, str) or not sdp.strip():
        raise VoiceLiveProtocolError("vendor answer has no SDP")
    # Sanity check only: an SDP answer always starts with the "v=" protocol version line.
    # A stray body that happens to carry a non-empty string here is not an answer.
    if not sdp.lstrip().startswith("v="):
        raise VoiceLiveProtocolError("vendor answer SDP is not an SDP")
    return payload


def _vendor_error_category(status: int) -> str:
    """A coarse category for logs — never the upstream body, which may echo key material."""
    if status in (401, 403):
        return "auth"
    if status == 404:
        return "not_found"
    if status == 429:
        return "rate_limited"
    if 500 <= status < 600:
        return "server_error"
    return "rejected"


def create_webrtc_session(sdp_offer: str, history: Optional[list] = None) -> dict[str, Any]:
    """Exchange the renderer's SDP offer for the Live session answer.

    Raises ``VoiceLiveMissingKeyError`` (no key for this profile),
    ``VoiceLiveDisabledError`` (``voice_chat_mode`` is not ``gpt-live``),
    ``VoiceLiveHistoryError`` (malformed ``history``), and ``VoiceLiveProtocolError`` /
    ``VoiceLiveUpstreamError`` for a rejected or unusable vendor answer. The upstream body is
    never logged, raised, or returned.
    """
    # Mode gate BEFORE credentials and before any network call: this mode is opt-in, so a
    # chained install must not start — or bill — a live session even when a key resolves.
    ensure_gpt_live_mode()
    live = _live_section()
    api_key, base_url = _resolve_credentials(live)
    if not api_key:
        raise VoiceLiveMissingKeyError(
            "GPT-Live needs an OpenAI API key (OPENAI_API_KEY or voice.gpt_live.api_key)"
        )
    body = json.dumps({
        "session": build_session_config(history),
        "transport": {"type": "webrtc", "sdp": sdp_offer},
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/live/sessions", data=body, method="POST",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        # Log status + category only. The upstream body is untrusted and can echo the key,
        # so it is neither logged nor put in the exception the route logs.
        category = _vendor_error_category(int(exc.code))
        logger.warning("GPT-Live session creation rejected: status=%s category=%s", exc.code, category)
        raise VoiceLiveUpstreamError(
            f"GPT-Live session creation rejected (status {exc.code}, {category})"
        ) from None
    return _validate_session_answer(payload)


def _normalize_vendor_session_id(session_id: str) -> str:
    """Validate a vendor live session id before it is embedded in a URL path."""
    vendor_id = str(session_id or "").strip()
    if not vendor_id:
        raise ValueError("session_id is required")
    if LIVE_VENDOR_SESSION_ID_RE.fullmatch(vendor_id) is None:
        raise ValueError(
            "session_id must match [A-Za-z0-9_-]{1,128} (vendor live session id)"
        )
    return vendor_id


def _vendor_session_hangup_url(base_url: str, session_id: str) -> str:
    vendor_id = _normalize_vendor_session_id(session_id)
    segment = quote(vendor_id, safe="")
    return f"{base_url.rstrip('/')}/live/sessions/{segment}/hangup"


def close_live_session(session_id: str) -> dict[str, Any]:
    """End a vendor Live session server-side.

    OpenAI documents ``POST /v1/live/sessions/{session_id}/hangup`` for SIP and WebRTC
    sessions (empty body, ``200 OK`` on success). WebRTC clients may also send
    ``session.close`` on the data channel; this HTTP hangup is the server-side teardown
    hook Pantheon can call when the renderer only has the vendor session id.

    Returns ``{"closed": True, "method": "hangup"}`` on success. Never logs or returns
    the upstream body.
    """
    ensure_gpt_live_mode()

    live = _live_section()
    api_key, base_url = _resolve_credentials(live)
    if not api_key:
        raise VoiceLiveMissingKeyError(
            "GPT-Live needs an OpenAI API key (OPENAI_API_KEY or voice.gpt_live.api_key)"
        )

    req = urllib.request.Request(
        _vendor_session_hangup_url(base_url, session_id),
        data=b"",
        method="POST",
        headers={"Authorization": f"Bearer {api_key}", "Accept": "*/*"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            status = int(getattr(resp, "status", 200) or 200)
    except urllib.error.HTTPError as exc:
        category = _vendor_error_category(int(exc.code))
        logger.warning(
            "GPT-Live session hangup rejected: status=%s category=%s",
            exc.code,
            category,
        )
        raise VoiceLiveUpstreamError(
            f"GPT-Live session hangup rejected (status {exc.code}, {category})"
        ) from None
    if status >= 400:
        category = _vendor_error_category(status)
        logger.warning(
            "GPT-Live session hangup rejected: status=%s category=%s",
            status,
            category,
        )
        raise VoiceLiveUpstreamError(
            f"GPT-Live session hangup rejected (status {status}, {category})"
        )
    return {"closed": True, "method": "hangup"}
