"""Per-session event sequencing + bounded replay for WS reconnects.

Every gateway event frame that flows through :func:`server.write_json` (and
therefore ``_emit``) is stamped with a per-session monotonic ``seq`` and
appended to a small ring buffer keyed by session id. A reconnecting client
calls the ``session.events.since`` RPC with its last observed seq; the server
replays everything newer from the buffer, then live events resume seamlessly.

Design constraints honored:
- stdio TUI path unaffected: frames gain a ``seq`` field only on event frames;
  Ink ignores unknown params keys.
- Thread safety: a single module lock guards counters + buffers; write_json
  already serializes per-transport writes, so stamping under the lock cannot
  reorder frames relative to each other.
- Memory bound: _REPLAY_BUFFER_MAX events / _REPLAY_SESSIONS_MAX session
  buffers, oldest session buffer evicted FIFO.
- Monotonic numbering: a session's seq counter OUTLIVES its buffer. Evicting
  the buffer must not reset the counter, or the next event for that session
  re-starts at seq 1 under the same epoch — a client holding a high watermark
  would filter the new event out as "already seen" and silently miss it.
  Counters are therefore bounded separately (_REPLAY_COUNTERS_MAX, oldest
  evicted FIFO). If a counter itself must be dropped, the numbering for that
  session restarts — unavoidable at that point — so the epoch is rotated to
  make the reset observable to clients (they compare the epoch and reset
  their watermarks on mismatch).
"""

from __future__ import annotations

import threading
import uuid
from collections import OrderedDict, deque

# Process identity for the replay contract. Seq counters live in-process, so
# a gateway restart silently resets them to 1 while clients still hold high
# watermarks — events_since(sid, 97) then returns [] with truncated=False and
# the client believes it missed nothing (and its stale watermark makes every
# future replay empty too). The epoch lets clients detect the restart and
# reset their watermarks.
#
# The epoch is also rotated when a seq counter is evicted (see
# _REPLAY_COUNTERS_MAX): that is the same "numbering reset underneath the
# client" situation, just reached by eviction instead of by restart.
_REPLAY_EPOCH = uuid.uuid4().hex

# Replay ring per session. A long turn emits ~hundreds of token events; this
# covers several minutes of streaming plus all control events.
_REPLAY_BUFFER_MAX = 512
# Distinct sessions remembered. Desktop users rarely exceed a dozen live chats.
_REPLAY_SESSIONS_MAX = 64
# Distinct seq counters remembered. Counters are tiny but unbounded session
# ids would leak, so they get their own (much larger) cap. 4096 sessions is
# far more than any real client holds at once, so a counter is only ever
# dropped long after its session is gone.
_REPLAY_COUNTERS_MAX = 4096

_replay_lock = threading.Lock()
# sid -> deque of (seq, event_object) where event_object is the frame's
# ``params`` dict (bare event: type/session_id/seq/payload) — the exact shape
# the client's dispatch path consumes.
_replay_buffers: "OrderedDict[str, deque]" = OrderedDict()
# sid -> highest seq handed out so far. OrderedDict (insertion-ordered) so
# the oldest counter can be evicted FIFO once _REPLAY_COUNTERS_MAX is hit.
# Kept independent of _replay_buffers: a buffer eviction must not reset a
# session's numbering.
_replay_next_seq: "OrderedDict[str, int]" = OrderedDict()


def replay_epoch() -> str:
    """Opaque token identifying this server process's seq numbering.

    Rotated when the process restarts (counters reset) and when a seq
    counter is evicted (that session's numbering restarts). Clients compare
    it against the epoch they learned at ``gateway.ready`` and reset their
    watermarks when it changes — that is the only signal that a numbering
    reset happened underneath them.
    """
    return _REPLAY_EPOCH


def _bound_counters() -> None:
    """Evict seq counters down to _REPLAY_COUNTERS_MAX, oldest first.

    Dropping a counter means that session's numbering restarts at 1, which a
    client holding a high watermark cannot detect on its own. So the epoch is
    rotated whenever a counter is lost: clients see the epoch change and reset
    their watermarks instead of silently dropping the session's new events.

    The evicted session's buffer is dropped too, so no buffer can outlive the
    counter that numbered it (a buffer holding seq 1..N next to a counter
    that restarts at 1 would replay duplicate seqs).
    Caller must hold _replay_lock.
    """
    global _REPLAY_EPOCH

    evicted = False
    while len(_replay_next_seq) > _REPLAY_COUNTERS_MAX:
        evicted_sid, _seq = _replay_next_seq.popitem(last=False)
        _replay_buffers.pop(evicted_sid, None)
        evicted = True

    if evicted:
        _REPLAY_EPOCH = uuid.uuid4().hex


def _stamp_event(obj: dict) -> None:
    """Stamp one outgoing event frame (mutates obj in place) and record it."""
    if obj.get("method") != "event":
        return
    params = obj.get("params")
    if not isinstance(params, dict):
        return
    sid = params.get("session_id") or ""
    if not sid:
        # Session-less global events (skin.changed etc.) are re-fetchable via
        # their own RPCs; no replay contract for them.
        return
    with _replay_lock:
        seq = _replay_next_seq.get(sid, 0) + 1
        _replay_next_seq[sid] = seq
        params["seq"] = seq
        buf = _replay_buffers.get(sid)
        if buf is None:
            buf = deque(maxlen=_REPLAY_BUFFER_MAX)
            _replay_buffers[sid] = buf
            while len(_replay_buffers) > _REPLAY_SESSIONS_MAX:
                # Evict the oldest session's BUFFER only. Its seq counter is
                # deliberately kept: if that session emits again it must
                # continue at seq+1, not restart at 1.
                _replay_buffers.popitem(last=False)
        buf.append((seq, params))
        _bound_counters()


def events_since(sid: str, last_seen: int) -> list[dict]:
    """Return recorded EVENT OBJECTS with seq > last_seen for *sid*, in order.

    Shape contract: each element is the frame's ``params`` dict — a bare event
    object with top-level ``type`` / ``session_id`` / ``seq`` — because that is
    exactly what the client's dispatch path consumes. Returning the full
    JSON-RPC envelope here would make every replayed event fail the client's
    ``event.type`` gate and be silently dropped.
    """
    with _replay_lock:
        buf = _replay_buffers.get(sid or "")
        if not buf:
            return []
        return [event for seq, event in buf if seq > last_seen]


def is_truncated(sid: str, last_seen: int) -> bool:
    """True when events between *last_seen* and the oldest retained seq were
    evicted — the client must refetch history instead of trusting the
    replay to be gap-free.

    Two eviction kinds leave a gap:
    - the session's ring overflowed → the gap sits before the buffer's
      oldest retained seq;
    - the session's buffer was evicted entirely (too many live sessions) →
      everything the client has not seen is gone. The seq counter survives
      the buffer eviction, so ``last_seen < latest_seq`` is the tell: the
      client is behind, and its replay window no longer exists.
    """
    with _replay_lock:
        sid = sid or ""
        buf = _replay_buffers.get(sid)
        if not buf:
            # No buffer: either the session is unknown (nothing was ever
            # evicted for it) or its buffer was evicted. A surviving counter
            # means events were dropped.
            return last_seen < _replay_next_seq.get(sid, 0)
        return last_seen + 1 < buf[0][0]


def latest_seq(sid: str) -> int:
    """Current highest stamped seq for *sid* (0 when unknown)."""
    with _replay_lock:
        return _replay_next_seq.get(sid or "", 0)


def reset_replay_state() -> None:
    """Test hook: drop every buffer AND every seq counter.

    Counters go too — a surviving counter would keep numbering high across a
    reset, while a dropped one restarts numbering at 1. The epoch is left
    alone: this hook is test-only and never runs in a live gateway.
    """
    with _replay_lock:
        _replay_buffers.clear()
        _replay_next_seq.clear()


def replay_stats() -> dict:
    """Telemetry: buffer occupancy for the ops/debug surface."""
    with _replay_lock:
        return {
            "sessions": len(_replay_buffers),
            "events": sum(len(b) for b in _replay_buffers.values()),
            "max_per_session": _REPLAY_BUFFER_MAX,
            # Seq counters outlive their buffers, so they are counted
            # separately — a counter count near the cap is the early warning
            # that an epoch rotation is coming.
            "counters": len(_replay_next_seq),
            "max_counters": _REPLAY_COUNTERS_MAX,
        }
