import asyncio
import threading
import time

import pytest

from hermes_cli.pty_session import RingBuffer


def test_ringbuffer_keeps_everything_under_capacity():
    rb = RingBuffer(10)
    rb.append(b"abc")
    rb.append(b"def")
    assert rb.snapshot() == b"abcdef"
    assert rb.truncated is False


def test_ringbuffer_drops_oldest_over_capacity():
    rb = RingBuffer(4)
    rb.append(b"abcdef")          # 6 bytes into a 4-byte buffer
    assert rb.snapshot() == b"cdef"
    assert rb.truncated is True




class FakeBridge:
    """Implements the bridge contract PtySession depends on."""

    def __init__(self, chunks):
        self._chunks = list(chunks)   # bytes; b"" = idle tick; None = EOF
        self.written = bytearray()
        self.closed = False
        self.resized = None

    def read(self, timeout):
        if not self._chunks:
            return b""                # idle
        return self._chunks.pop(0)

    def write(self, data):
        self.written.extend(data)

    def resize(self, cols, rows):
        self.resized = (cols, rows)

    def close(self):
        self.closed = True


class FakeWS:
    def __init__(self):
        self.sent = []               # list of ("bytes"|"text", payload)
        self.close_code = None

    async def send_bytes(self, data):
        self.sent.append(("bytes", bytes(data)))

    async def send_text(self, text):
        self.sent.append(("text", text))

    async def close(self, code=1000, reason=""):
        self.close_code = code


@pytest.mark.asyncio
async def test_attach_replays_buffer_then_streams_live():
    from hermes_cli.pty_session import PtySession
    bridge = FakeBridge([b"hello ", b"world", None])
    s = PtySession("k", bridge, buffer_cap=1024, read_timeout=0.01)
    await s.start()
    await asyncio.sleep(0.05)                      # drain consumes "hello world"
    ws = FakeWS()
    await s.attach(ws)
    replay = b"".join(p for kind, p in ws.sent if kind == "bytes")
    assert replay == b"hello world"
    await s.close()


@pytest.mark.asyncio
async def test_reattach_can_force_complete_tui_redraw_after_replay():
    """A fresh terminal cannot reconstruct a differential ANSI tail alone."""
    from hermes_cli.pty_session import PtySession

    bridge = FakeBridge([b"partial differential frame", b""])
    s = PtySession("k", bridge, buffer_cap=1024, read_timeout=0.01)
    await s.start()
    await asyncio.sleep(0.05)

    ws = FakeWS()
    await s.attach(ws, force_redraw=True)

    replay = b"".join(p for kind, p in ws.sent if kind == "bytes")
    assert replay == b"partial differential frame"
    assert bytes(bridge.written) == b"\x0c"
    await s.close()


@pytest.mark.asyncio
async def test_detach_keeps_draining_into_buffer():
    from hermes_cli.pty_session import PtySession
    bridge = FakeBridge([b"one", b"", b"two"])
    s = PtySession("k", bridge, buffer_cap=1024, read_timeout=0.01)
    await s.start()
    ws = FakeWS()
    await s.attach(ws)
    s.detach(ws)
    assert s.attached is False
    assert s.last_detached_at is not None
    await asyncio.sleep(0.05)                      # "two" drains while detached
    ws2 = FakeWS()
    await s.attach(ws2)
    replay = b"".join(p for kind, p in ws2.sent if kind == "bytes")
    assert replay == b"onetwo"
    await s.close()


@pytest.mark.asyncio
async def test_eof_marks_dead_and_closes_socket_4410():
    from hermes_cli.pty_session import PtySession
    bridge = FakeBridge([b"bye", None])
    s = PtySession("k", bridge, buffer_cap=1024, read_timeout=0.01)
    await s.start()
    ws = FakeWS()
    await s.attach(ws)
    await asyncio.sleep(0.05)                      # drain hits None (EOF)
    assert s.alive is False
    assert ws.close_code == 4410
    await s.close()


from hermes_cli.pty_session import PtySessionRegistry, RegistryFull


def make_registry(ttl=1800.0, max_sessions=16):
    return PtySessionRegistry(ttl=ttl, max_sessions=max_sessions,
                              buffer_cap=1024, read_timeout=0.01)


@pytest.mark.asyncio
async def test_same_key_reattaches_same_session():
    reg = make_registry()
    b1 = FakeBridge([b"", b"", b""])
    s1, created1 = await reg.attach_or_spawn("tok", spawn=lambda: b1)
    s2, created2 = await reg.attach_or_spawn("tok", spawn=lambda: FakeBridge([]))
    assert created1 is True and created2 is False
    assert s1 is s2
    assert s2.bridge is b1                     # second spawn callable was NOT used
    await reg.close_all()


@pytest.mark.asyncio
async def test_concurrent_same_key_spawns_once():
    """Two simultaneous attach_or_spawn calls must not spawn duplicate PTYs."""
    reg = make_registry()
    spawn_count = {"n": 0}
    started = threading.Event()
    proceed = threading.Event()
    bridges = []

    def blocking_spawn():
        spawn_count["n"] += 1
        started.set()
        proceed.wait(timeout=5.0)
        b = FakeBridge([b"", b""])
        bridges.append(b)
        return b

    t1 = asyncio.create_task(reg.attach_or_spawn("tok", spawn=blocking_spawn))
    t2 = None
    try:
        for _ in range(500):
            if started.is_set():
                break
            await asyncio.sleep(0.001)
        assert started.is_set()
        t2 = asyncio.create_task(reg.attach_or_spawn("tok", spawn=blocking_spawn))
        proceed.set()
        (s1, c1), (s2, c2) = await asyncio.gather(t1, t2)
        t1 = t2 = None
        assert spawn_count["n"] == 1
        assert s1 is s2
        assert (c1, c2).count(True) == 1
    finally:
        proceed.set()
        for t in (t1, t2):
            if t is not None and not t.done():
                t.cancel()
                try:
                    await t
                except (asyncio.CancelledError, Exception):
                    pass
        await reg.close_all()
        assert all(b.closed for b in bridges)


@pytest.mark.asyncio
async def test_cancel_during_spawn_closes_bridge_and_same_key_retry_is_single():
    """Cancelling the spawn owner must not leak a PTY or allow duplicates."""
    reg = make_registry()
    bridges = []
    started = threading.Event()
    proceed = threading.Event()

    def blocking_spawn():
        b = FakeBridge([b"", b""])
        bridges.append(b)
        started.set()
        proceed.wait(timeout=5.0)
        return b

    owner = asyncio.create_task(reg.attach_or_spawn("tok", spawn=blocking_spawn))
    try:
        for _ in range(500):
            if started.is_set():
                break
            await asyncio.sleep(0.001)
        assert started.is_set()

        owner.cancel()
        # Unblock the worker so the shielded spawn can finish; the cancelled
        # owner must reclaim and close that bridge before releasing the lock.
        proceed.set()
        with pytest.raises(asyncio.CancelledError):
            await owner
        owner = None

        assert len(bridges) == 1
        assert bridges[0].closed is True
        assert "tok" not in reg._sessions
        assert "tok" not in reg._key_locks

        def retry_spawn():
            b = FakeBridge([b"", b""])
            bridges.append(b)
            return b

        session, created = await reg.attach_or_spawn("tok", spawn=retry_spawn)
        assert created is True
        assert session.bridge is bridges[1]
        assert bridges[1].closed is False
        assert list(reg._sessions) == ["tok"]
        assert sum(1 for b in bridges if not b.closed) == 1
    finally:
        proceed.set()
        if owner is not None and not owner.done():
            owner.cancel()
            try:
                await owner
            except (asyncio.CancelledError, Exception):
                pass
        await reg.close_all()
        assert all(b.closed for b in bridges)
        assert reg._key_locks == {}


@pytest.mark.asyncio
async def test_rejected_attach_keys_do_not_retain_locks():
    """Capacity rejects and failed spawns must not grow _key_locks forever."""
    reg = make_registry(max_sessions=0)
    for i in range(50):
        with pytest.raises(RegistryFull):
            await reg.attach_or_spawn(f"reject-{i}", spawn=lambda: FakeBridge([]))
    assert reg._sessions == {}
    assert reg._key_locks == {}

    reg2 = make_registry(max_sessions=4)

    def boom():
        raise RuntimeError("spawn failed")

    for i in range(20):
        with pytest.raises(RuntimeError, match="spawn failed"):
            await reg2.attach_or_spawn(f"fail-{i}", spawn=boom)
    assert reg2._sessions == {}
    assert reg2._key_locks == {}


@pytest.mark.asyncio
async def test_reap_and_close_all_prune_key_locks():
    reg = make_registry(ttl=0.01)
    b = FakeBridge([None])  # EOF on first read → session dies
    s, _ = await reg.attach_or_spawn("tok", spawn=lambda: b)
    ws = FakeWS()
    await s.attach(ws)
    await asyncio.sleep(0.05)  # drain hits EOF
    assert s.alive is False
    await reg.reap_idle()
    assert "tok" not in reg._sessions
    assert "tok" not in reg._key_locks

    b2 = FakeBridge([b"", b""])
    await reg.attach_or_spawn("keep", spawn=lambda: b2)
    assert "keep" in reg._key_locks
    await reg.close_all()
    assert reg._sessions == {}
    assert reg._key_locks == {}


@pytest.mark.asyncio
async def test_close_all_clears_stale_locks_for_loop_reuse():
    """Module-level registry can outlive a lifespan; leftover locks must go."""
    reg = make_registry()
    reg._key_locks["stale"] = asyncio.Lock()
    await reg.close_all()
    assert reg._key_locks == {}
    b = FakeBridge([b"", b""])
    try:
        s, created = await reg.attach_or_spawn("stale", spawn=lambda: b)
        assert created is True
        assert s.bridge is b
    finally:
        await reg.close_all()
    assert reg._key_locks == {}


@pytest.mark.asyncio
async def test_new_key_at_capacity_raises_when_none_reapable():
    reg = make_registry(max_sessions=1)
    b = FakeBridge([b"", b""])
    try:
        s, _ = await reg.attach_or_spawn("a", spawn=lambda: b)
        await s.attach(FakeWS())                    # attached → not reapable
        with pytest.raises(RegistryFull):
            await reg.attach_or_spawn("b", spawn=lambda: FakeBridge([]))
        # Rejected key must not leave a permanent lock entry.
        assert "b" not in reg._key_locks
    finally:
        await reg.close_all()


@pytest.mark.asyncio
async def test_reaper_loop_invokes_reap(monkeypatch):
    from hermes_cli.pty_session import run_reaper
    reg = make_registry()
    calls = {"n": 0}

    async def fake_reap(now=None):
        calls["n"] += 1

    monkeypatch.setattr(reg, "reap_idle", fake_reap)
    task = asyncio.create_task(run_reaper(reg, interval=0.01))
    try:
        await asyncio.sleep(0.05)
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
    assert calls["n"] >= 2
