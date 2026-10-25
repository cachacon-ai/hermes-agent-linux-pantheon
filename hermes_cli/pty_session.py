"""Keep-alive PTY sessions for dashboard terminals.

A PTY process outlives the WebSocket that created it: a single drain task
always reads the PTY into a bounded RingBuffer and forwards to the attached
socket when present. Reconnecting with the same opaque token replays the
buffer and resumes live. See
docs/superpowers/specs/2026-06-20-pty-keepalive-reattach-design.md.
"""
from __future__ import annotations

import asyncio
import time
from typing import Optional

WS_CLOSE_PROCESS_EXITED = 4410
WS_CLOSE_SUPERSEDED = 4409
TUI_FORCE_REDRAW = b"\x0c"


class RingBuffer:
    """Keeps only the most recent ``capacity`` bytes appended to it."""

    def __init__(self, capacity: int) -> None:
        self._cap = capacity
        self._buf = bytearray()
        self._truncated = False

    def append(self, data: bytes) -> None:
        self._buf.extend(data)
        overflow = len(self._buf) - self._cap
        if overflow > 0:
            del self._buf[:overflow]
            self._truncated = True

    def snapshot(self) -> bytes:
        return bytes(self._buf)

    @property
    def truncated(self) -> bool:
        return self._truncated


class PtySession:
    def __init__(self, key: str, bridge, *, buffer_cap: int, read_timeout: float) -> None:
        self.key = key
        self.bridge = bridge
        self.buffer = RingBuffer(buffer_cap)
        self.alive = True
        self.attached = False
        self.last_detached_at: Optional[float] = None
        self._read_timeout = read_timeout
        self._ws = None
        self._drain_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        self._drain_task = asyncio.create_task(self._drain())

    async def _drain(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            chunk = await loop.run_in_executor(None, self.bridge.read, self._read_timeout)
            if chunk is None:                       # EOF — the agent process exited
                self.alive = False
                ws = self._ws
                if ws is not None:
                    try:
                        await ws.close(code=WS_CLOSE_PROCESS_EXITED)
                    except Exception:
                        pass
                return
            if not chunk:                            # idle tick
                await asyncio.sleep(0)
                continue
            self.buffer.append(chunk)
            ws = self._ws
            if ws is not None:
                try:
                    await ws.send_bytes(chunk)
                except Exception:
                    pass                             # detached mid-send; keep buffering

    async def attach(self, ws, *, force_redraw: bool = False) -> None:
        """Attach a browser terminal and replay buffered PTY output.

        The TUI uses an alternate screen and differential rendering, so a
        bounded ANSI tail is not guaranteed to be a self-contained frame.
        Reattaching a fresh xterm therefore asks the live TUI to emit one
        complete redraw after the replay.
        """
        old = self._ws
        if old is not None and old is not ws:
            try:
                await old.close(code=WS_CLOSE_SUPERSEDED)
            except Exception:
                pass
        self._ws = ws
        self.attached = True
        self.last_detached_at = None
        snap = self.buffer.snapshot()
        if snap:
            await ws.send_bytes(snap)
        if force_redraw:
            self.bridge.write(TUI_FORCE_REDRAW)

    def detach(self, ws) -> None:
        # Only the currently-attached socket may mark the session detached.
        # A superseded socket's handler also calls detach on its way out
        # (its ``finally`` runs after the new tab attached); flipping
        # ``attached`` then would make a session with a live viewer look
        # idle and reapable.
        if self._ws is not ws:
            return
        self._ws = None
        self.attached = False
        self.last_detached_at = time.monotonic()

    async def close(self) -> None:
        if self._drain_task is not None:
            self._drain_task.cancel()
            try:
                await self._drain_task
            except (asyncio.CancelledError, Exception):
                pass
        try:
            # bridge.close() joins the child — blocking; keep it off the
            # event loop (#53227).
            await asyncio.to_thread(self.bridge.close)
        except Exception:
            pass


from typing import Callable, Dict, Tuple


class RegistryFull(Exception):
    pass


async def run_reaper(registry: "PtySessionRegistry", *, interval: float = 60.0) -> None:
    """Periodically reap idle/dead keep-alive sessions. Cancelled on shutdown."""
    while True:
        await asyncio.sleep(interval)
        try:
            await registry.reap_idle()
        except Exception:
            pass


class PtySessionRegistry:
    def __init__(self, *, ttl: float, max_sessions: int,
                 buffer_cap: int, read_timeout: float) -> None:
        self._ttl = ttl
        self._max = max_sessions
        self._buffer_cap = buffer_cap
        self._read_timeout = read_timeout
        self._sessions: Dict[str, PtySession] = {}
        self._key_locks: Dict[str, asyncio.Lock] = {}
        # Tasks currently inside attach_or_spawn for a key (asyncio is
        # single-threaded, so this counter is mutated only between awaits).
        self._lock_refs: Dict[str, int] = {}

    def _acquire_key_lock(self, key: str) -> asyncio.Lock:
        lock = self._key_locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            self._key_locks[key] = lock
        self._lock_refs[key] = self._lock_refs.get(key, 0) + 1
        return lock

    def _release_key_lock(self, key: str) -> None:
        """Drop the per-key lock once nothing can still reference it."""
        refs = self._lock_refs.get(key, 0) - 1
        if refs > 0:
            self._lock_refs[key] = refs
            return
        self._lock_refs.pop(key, None)
        if key not in self._sessions:
            self._key_locks.pop(key, None)

    def _prune_key_lock(self, key: str) -> None:
        """Remove a key's lock when idle and unreferenced (reap/shutdown)."""
        if self._lock_refs.get(key, 0) > 0:
            return
        if key in self._sessions:
            return
        self._key_locks.pop(key, None)

    @staticmethod
    async def _close_bridge(bridge) -> None:
        """Close a bridge even when the calling task is being cancelled."""
        task = asyncio.current_task()
        cleared = 0
        if task is not None:
            # 3.11+: a pending cancel aborts every await, including cleanup.
            while task.cancelling() > 0:
                task.uncancel()
                cleared += 1
        try:
            try:
                await asyncio.shield(asyncio.to_thread(bridge.close))
            except Exception:
                pass
        finally:
            if task is not None:
                for _ in range(cleared):
                    task.cancel()

    async def _spawn_owned(self, spawn: Callable[[], object]):
        """Run ``spawn`` off-thread and own the bridge through cancellation.

        Cancelling an executor await does not stop the worker thread, so a
        bare await would release the per-key lock while the PTY keeps coming
        up unregistered. Shield the spawn, reclaim the result on cancel, and
        close any bridge that never gets registered.
        """
        loop = asyncio.get_running_loop()
        spawn_fut = loop.run_in_executor(None, spawn)
        try:
            return await asyncio.shield(spawn_fut)
        except asyncio.CancelledError:
            task = asyncio.current_task()
            cleared = 0
            if task is not None:
                while task.cancelling() > 0:
                    task.uncancel()
                    cleared += 1
            bridge = None
            try:
                try:
                    bridge = await asyncio.shield(spawn_fut)
                except Exception:
                    bridge = None
                if bridge is not None:
                    # Already cleared cancel depth; close without nesting
                    # another uncancel/restore cycle.
                    try:
                        await asyncio.shield(asyncio.to_thread(bridge.close))
                    except Exception:
                        pass
                    bridge = None
            finally:
                if task is not None:
                    for _ in range(cleared):
                        task.cancel()
            raise

    async def attach_or_spawn(self, key: str, *, spawn: Callable[[], object]
                              ) -> Tuple[PtySession, bool]:
        # Concurrent /api/pty upgrades with the same attach token must not
        # both pass the "no session yet" check while spawn is in flight.
        lock = self._acquire_key_lock(key)
        try:
            async with lock:
                await self.reap_idle()
                existing = self._sessions.get(key)
                if existing is not None and existing.alive:
                    return existing, False
                if existing is not None:                       # dead remnant
                    await existing.close()
                    self._sessions.pop(key, None)
                    self._prune_key_lock(key)
                if len(self._sessions) >= self._max:
                    self._reap_one_idle_or_raise()
                # PTY spawn does blocking fork/exec work — keep it off the
                # event loop (#53227). Own the result through cancellation so
                # a cancelled owner cannot leak a live PTY outside the
                # registry / cap / reaper (PAN-81).
                bridge = await self._spawn_owned(spawn)
                session = PtySession(key, bridge, buffer_cap=self._buffer_cap,
                                     read_timeout=self._read_timeout)
                try:
                    await session.start()
                    self._sessions[key] = session
                except BaseException:
                    await self._close_bridge(bridge)
                    raise
                return session, True
        finally:
            self._release_key_lock(key)

    def detach(self, key: str, ws) -> None:
        s = self._sessions.get(key)
        if s is not None:
            s.detach(ws)

    async def reap_idle(self, now: Optional[float] = None) -> None:
        now = time.monotonic() if now is None else now
        doomed = [
            key for key, s in self._sessions.items()
            if (not s.alive)
            or (not s.attached and s.last_detached_at is not None
                and (now - s.last_detached_at) > self._ttl)
        ]
        for key in doomed:
            await self._sessions.pop(key).close()
            self._prune_key_lock(key)

    def _reap_one_idle_or_raise(self) -> None:
        idle = [s for s in self._sessions.values()
                if not s.attached and s.last_detached_at is not None]
        if not idle:
            raise RegistryFull()
        oldest = min(idle, key=lambda s: s.last_detached_at or 0.0)
        self._sessions.pop(oldest.key, None)
        self._prune_key_lock(oldest.key)
        asyncio.create_task(oldest.close())

    async def close_all(self) -> None:
        for key in list(self._sessions):
            await self._sessions.pop(key).close()
            self._prune_key_lock(key)
        # Drop any leftover locks from rejected/cancelled attaches that had
        # no in-flight waiter left. Locks still referenced by an in-flight
        # attach_or_spawn stay until that caller's finally releases them.
        for key in list(self._key_locks):
            self._prune_key_lock(key)
