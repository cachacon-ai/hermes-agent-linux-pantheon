"""Tests for tui_gateway.event_replay — per-session event seq + replay ring."""

import threading

import pytest

from tui_gateway import event_replay
from tui_gateway.event_replay import (
    latest_seq,
    reset_replay_state,
    events_since,
    replay_epoch,
    replay_stats,
)


@pytest.fixture(autouse=True)
def _clean():
    reset_replay_state()
    yield
    reset_replay_state()


def _frame(sid, etype="message.delta"):
    return {
        "jsonrpc": "2.0",
        "method": "event",
        "params": {"type": etype, "session_id": sid, "payload": {}},
    }


def test_stamp_adds_monotonic_seq_per_session():
    f1 = _frame("s1")
    f2 = _frame("s1")
    other = _frame("s2")

    event_replay._stamp_event(f1)
    event_replay._stamp_event(other)
    event_replay._stamp_event(f2)

    assert f1["params"]["seq"] == 1
    assert f2["params"]["seq"] == 2  # per-session counter, unaffected by s2
    assert other["params"]["seq"] == 1


def test_stamp_ignores_non_event_and_sessionless_frames():
    rpc = {"jsonrpc": "2.0", "id": 1, "result": {}}
    no_sid = {"jsonrpc": "2.0", "method": "event", "params": {"type": "skin.changed"}}

    event_replay._stamp_event(rpc)
    event_replay._stamp_event(no_sid)

    assert "seq" not in rpc
    assert "seq" not in no_sid["params"]
    assert replay_stats()["events"] == 0


def test_events_since_returns_only_newer_frames_in_order():
    frames = [_frame("s1") for _ in range(5)]
    for f in frames:
        event_replay._stamp_event(f)

    got = events_since("s1", 3)
    assert [e["seq"] for e in got] == [4, 5]
    assert events_since("s1", 0) == [f["params"] for f in frames]
    assert events_since("s1", 99) == []
    assert latest_seq("s1") == 5


def test_events_since_returns_client_dispatchable_event_objects():
    """Cross-language contract: the client's replay loop dispatches an element
    only when it has a TOP-LEVEL ``type`` (json-rpc-gateway.ts fetchReplay:
    ``if (!event?.type) continue``). Returning full JSON-RPC envelopes here
    makes every replayed event silently droppable — the original #94219 bug.
    """
    event_replay._stamp_event(_frame("s1"))
    (event,) = events_since("s1", 0)

    # Bare event object, not an envelope.
    assert event["type"] == "message.delta"
    assert event["session_id"] == "s1"
    assert event["seq"] == 1
    assert "jsonrpc" not in event
    assert "method" not in event
    assert "params" not in event


def test_run_link_correlation_survives_the_gateway_replay_ring():
    frame = _frame("s1", "run.link")
    frame["params"].update(
        run_id="active-run",
        submission_id="correction-submission",
        source_submission_id="correction-submission",
        payload={"relation": "steered"},
    )
    event_replay._stamp_event(frame)

    (event,) = events_since("s1", 0)
    assert event["type"] == "run.link"
    assert event["run_id"] == "active-run"
    assert event["submission_id"] == "correction-submission"
    assert event["source_submission_id"] == "correction-submission"
    assert event["payload"] == {"relation": "steered"}


def test_unknown_session_returns_empty():
    assert events_since("nope", 0) == []
    assert latest_seq("nope") == 0


def test_ring_buffer_is_bounded():
    for i in range(event_replay._REPLAY_BUFFER_MAX + 50):
        event_replay._stamp_event(_frame("s1"))

    stats = replay_stats()
    assert stats["events"] == event_replay._REPLAY_BUFFER_MAX
    # Oldest evicted: last_seen=0 must report truncation via the RPC contract.
    buf = event_replay._replay_buffers["s1"]
    assert buf[0][0] > 1


def test_session_count_bounded_with_fifo_eviction():
    for i in range(event_replay._REPLAY_SESSIONS_MAX + 10):
        event_replay._stamp_event(_frame(f"s{i}"))

    stats = replay_stats()
    assert stats["sessions"] == event_replay._REPLAY_SESSIONS_MAX
    assert events_since("s0", 0) == []  # oldest session fully evicted
    # The counter survived the buffer eviction, so the client behind it must
    # be told it missed events — empty replay AND truncated=true.
    assert latest_seq("s0") > 0
    assert event_replay.is_truncated("s0", 0) is True
    assert latest_seq(f"s{event_replay._REPLAY_SESSIONS_MAX + 9}") == 1


def test_evicted_session_keeps_monotonic_seq():
    """Regression: evicting a session's BUFFER must not reset its seq counter.

    The counter used to be popped alongside the buffer, so the session's next
    event restarted at seq 1 under the same epoch. A client holding a high
    watermark filtered that event out as "already seen" and silently dropped
    it — no gap, no error, just missing events.
    """
    event_replay._stamp_event(_frame("s1"))
    event_replay._stamp_event(_frame("s1"))
    old_seq = latest_seq("s1")
    assert old_seq == 2

    # Overflow the session table so s1's buffer is evicted.
    for i in range(event_replay._REPLAY_SESSIONS_MAX + 2):
        event_replay._stamp_event(_frame(f"filler{i}"))

    assert "s1" not in event_replay._replay_buffers  # buffer gone
    assert latest_seq("s1") == old_seq  # counter survives the eviction

    again = _frame("s1")
    event_replay._stamp_event(again)

    assert again["params"]["seq"] == old_seq + 1  # monotonic, NOT 1
    # And the client can actually replay it from its watermark.
    assert [e["seq"] for e in events_since("s1", old_seq)] == [old_seq + 1]


def test_evicted_session_reports_truncation():
    """A client behind an evicted session must be told it missed events."""
    for i in range(event_replay._REPLAY_SESSIONS_MAX + 2):
        event_replay._stamp_event(_frame(f"s{i}"))

    assert "s0" not in event_replay._replay_buffers  # buffer evicted
    latest = latest_seq("s0")
    assert latest == 1  # counter survived, so we know seq 1 was issued

    # Client is behind (missed seq 1) → the event is gone → truncated.
    assert event_replay.is_truncated("s0", 0) is True
    # Client is caught up → nothing missing.
    assert event_replay.is_truncated("s0", latest) is False
    # Totally unknown session, nothing ever evicted → not truncated.
    assert event_replay.is_truncated("nope", 0) is False


def test_buffer_eviction_alone_does_not_rotate_epoch():
    """Only a lost COUNTER restarts numbering, so only that rotates the epoch."""
    before = replay_epoch()

    for i in range(event_replay._REPLAY_SESSIONS_MAX + 2):
        event_replay._stamp_event(_frame(f"s{i}"))

    assert replay_epoch() == before
    # Counters are retained, so they outnumber the retained buffers.
    assert len(event_replay._replay_next_seq) > len(event_replay._replay_buffers)


def test_counter_eviction_rotates_epoch(monkeypatch):
    """Losing a counter restarts that session's numbering → rotate the epoch.

    Clients compare the epoch (gateway.ready / session.events.since) and reset
    their watermarks when it changes. Without the rotation a client would keep
    a stale high watermark and silently drop the restarted session's events.
    """
    monkeypatch.setattr(event_replay, "_REPLAY_COUNTERS_MAX", 2)
    before = replay_epoch()

    for sid in ("s1", "s2", "s3"):
        event_replay._stamp_event(_frame(sid))

    # Oldest counter dropped (and its buffer with it, so no buffer can
    # outlive the counter that numbered it).
    assert latest_seq("s1") == 0
    assert "s1" not in event_replay._replay_buffers
    # Survivors keep their numbering.
    assert latest_seq("s2") == 1
    assert latest_seq("s3") == 1
    # Epoch rotated → clients can detect the reset.
    assert replay_epoch() != before


def test_replay_epoch_function_reflects_rotation(monkeypatch):
    """``from tui_gateway.event_replay import replay_epoch`` (tui_gateway/entry.py,
    tui_gateway/ws.py) must observe the CURRENT epoch, not the import-time one."""
    monkeypatch.setattr(event_replay, "_REPLAY_COUNTERS_MAX", 1)
    before = replay_epoch()

    event_replay._stamp_event(_frame("s1"))
    event_replay._stamp_event(_frame("s2"))  # evicts s1's counter → rotate

    assert replay_epoch() != before
    assert event_replay.replay_epoch() == replay_epoch()


def test_counters_are_bounded(monkeypatch):
    monkeypatch.setattr(event_replay, "_REPLAY_COUNTERS_MAX", 3)

    for i in range(20):
        event_replay._stamp_event(_frame(f"s{i}"))

    assert len(event_replay._replay_next_seq) <= 3
    assert replay_stats()["counters"] == len(event_replay._replay_next_seq)
    # Newest sessions keep their numbering; the oldest ones are dropped.
    assert latest_seq("s19") == 1
    assert latest_seq("s0") == 0


def test_counter_eviction_is_lru_not_first_seen(monkeypatch):
    """Re-stamping a session must refresh its position in the counter table.

    The counter table is capped at _REPLAY_COUNTERS_MAX. With first-seen
    ordering, a long-lived session (seen early, still emitting) gets evicted
    by a burst of brand-new ids, and its numbering restarts behind an epoch
    rotation. LRU keeps the counter of the session that is actually in use.
    """
    monkeypatch.setattr(event_replay, "_REPLAY_COUNTERS_MAX", 3)

    event_replay._stamp_event(_frame("s1"))
    event_replay._stamp_event(_frame("s2"))
    event_replay._stamp_event(_frame("s3"))
    # s1 emits again → it is the most recently used counter now.
    event_replay._stamp_event(_frame("s1"))
    assert latest_seq("s1") == 2

    # A new id pushes the table over the cap → the LRU counter (s2) goes.
    event_replay._stamp_event(_frame("s4"))

    assert latest_seq("s2") == 0  # evicted: least recently used
    assert latest_seq("s1") == 2  # survived: re-stamped, so recently used
    assert latest_seq("s3") == 1
    assert latest_seq("s4") == 1


def test_negative_last_seen_is_treated_as_zero():
    """A malformed negative watermark must not fake a truncation.

    ``last_seen < 0`` used to read as "the client is behind" for an unknown
    session (``-1 < 0``), so a garbage watermark produced truncated=true with
    an empty replay and the client refetched history for a session that never
    lost an event. Negative means "I have seen nothing" — exactly what 0 says.
    """
    # Unknown session: nothing was ever evicted → not truncated.
    assert event_replay.is_truncated("nope", -1) is False
    assert events_since("nope", -1) == []

    # Known session whose buffer starts at seq 1: the client gets every
    # event, so nothing is missing even with a garbage watermark.
    event_replay._stamp_event(_frame("s1"))
    assert event_replay.is_truncated("s1", -1) is False
    assert [e["seq"] for e in events_since("s1", -1)] == [1]

    # Evicted session: the client still missed seq 1 → truncated.
    for i in range(event_replay._REPLAY_SESSIONS_MAX + 2):
        event_replay._stamp_event(_frame(f"filler{i}"))

    assert "s1" not in event_replay._replay_buffers  # buffer evicted
    assert event_replay.is_truncated("s1", -1) is True


def test_concurrent_stamping_never_drops_or_duplicates_seq():
    errors = []

    def worker(sid):
        try:
            seen = set()
            for _ in range(200):
                f = _frame(sid)
                event_replay._stamp_event(f)
                seq = f["params"]["seq"]
                assert seq not in seen
                seen.add(seq)
        except AssertionError as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(f"t{i}",)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    assert replay_stats()["events"] == 8 * 200


def test_truncation_detection_semantics():
    """The RPC handler's truncated flag: gap between last_seen and buffer start."""
    # Overflow the ring so the oldest events are genuinely evicted.
    for _ in range(event_replay._REPLAY_BUFFER_MAX + 10):
        event_replay._stamp_event(_frame("s1"))

    with event_replay._replay_lock:
        oldest = event_replay._replay_buffers["s1"][0][0]

    assert oldest > 1  # eviction happened

    # Client saw everything up to just before the buffer → NOT truncated.
    assert not event_replay.is_truncated("s1", oldest - 1)
    # Client saw seq 5, buffer starts later → truncated.
    assert event_replay.is_truncated("s1", 5)
    # Unknown session: nothing evicted, nothing truncated.
    assert not event_replay.is_truncated("nope", 0)
