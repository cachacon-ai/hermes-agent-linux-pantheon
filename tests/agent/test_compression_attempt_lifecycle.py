"""Attempt-lifecycle regression tests for #97488 / #96775.

Pins the three attempt-level lifecycle guarantees added after PR #98628
collapsed lean compaction to one auxiliary request per attempt:

1. A ceiling/idle-timeout host TEARS DOWN its worker: a cooperative worker is
   joined within the bounded grace (releasing the lease normally), while an
   uninterruptible worker is orphaned behind the poison fence — its late
   result is discarded and, on the total-ceiling path, the durable lease is
   retained until it exits so no new attempt overlaps the unchanged session.
2. A failed/stalled/cancelled attempt records a durable per-session backoff
   (strategy + failure kind stamped into the state.db cooldown row) that
   SURVIVES a gateway restart; the next automatic turn skips the same
   strategy inside the window, and a successful compression clears it.
3. Late results from a superseded attempt are discarded, never committed
   over newer state (generation counter), and a transiently-blocked no-op is
   reported as a soft defer — never compression_exhausted (false auto-reset).
"""

from __future__ import annotations

import concurrent.futures
import copy
import logging
import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest

from agent.auxiliary_client import AuxiliaryExplicitCancellation
from agent.conversation_compression import (
    CompressionCommitFence,
    _claim_compressor_attempt,
    compress_context,
    compression_blocked_transiently,
    run_compress_context_with_progress_timeout,
)
from hermes_state import SessionDB


class _LateStartExecutor:
    """Executor whose job only starts once ``delay`` has elapsed.

    Models the shared compress pool dequeuing the job (so ``future.cancel()``
    is a no-op) but not executing its first bytecode until the total ceiling
    has already expired — the interleaving a loaded CI box produces when the
    cold-import + executor-creation + submit path costs more than the ceiling.
    Deterministic, so no scheduler luck is needed to hit the shape.
    """

    def __init__(self, delay: float) -> None:
        self._delay = delay

    def submit(self, fn, fence):  # noqa: ANN001 - mirrors ThreadPoolExecutor
        future = concurrent.futures.Future()
        time.sleep(self._delay)
        try:
            future.set_result(fn(fence))
        except BaseException as exc:  # noqa: BLE001 - mirrors the pool
            future.set_exception(exc)
        return future


class _PromptStartExecutor:
    """Real pool whose job is guaranteed to reach its first bytecode at once.

    The shared compress pool is process-wide and daemon: on a loaded CI box its
    worker thread can miss a sub-second window entirely, so
    ``_fence_gated_worker`` takes the F6 stale-job skip and the worker body never
    runs. That is CORRECT production behavior (it burns no LLM call), but it
    makes a test that is about the bounded-grace JOIN non-deterministic: 1 run in
    ~300 under GIL pressure, and it is indistinguishable from a missing join.

    ``submit`` blocks until the job has actually started, so the tests that pin
    the join are deterministic about WHICH path they exercise. Nothing in the
    compression module is replaced — only the scheduling is made prompt.
    """

    def __init__(self) -> None:
        self._pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self._started = threading.Event()
        # Spawn the worker thread NOW, in the caller's setup, so the attempt's
        # own ceiling budget is not charged for thread creation. Under GIL
        # pressure that spawn measures ~0.05-0.8s, which matters against the
        # 0.2-2.0s ceilings these tests use.
        self._pool.submit(lambda: None).result(timeout=30)

    def submit(self, fn, fence):  # noqa: ANN001 - mirrors ThreadPoolExecutor
        def run():
            self._started.set()
            return fn(fence)

        future = self._pool.submit(run)
        # Deterministic start: do not let the host begin its ceiling clock until
        # the job body is actually running.
        assert self._started.wait(timeout=5), "job never started"
        return future

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False)


@pytest.fixture(scope="module", autouse=True)
def _prime_compress_path():
    """Warm the process once per file, before any test arms a fence."""
    _prime_compress_path_once()


def _prime_compress_path_once() -> None:
    """Pay this PROCESS's one-time compression machinery cost once.

    Every attempt goes through ``propagate_context_to_thread``, whose
    ``_callback_api()`` lazily imports ``tools.thread_context`` and then
    ``tools.terminal_tool`` (tools/thread_context.py:43-61). That import is
    charged to whichever attempt pays it first, and on a cold, loaded machine it
    is enormous: measured 22.7s with 4 CPU-bound burner threads in a fresh
    process, 0.000s once warm.

    The budgets below (0.2s-4s ceilings) start at the fence, so a cold process
    can expire the deadline before the pool thread reaches its first bytecode —
    ``_fence_gated_worker`` then takes the F6 stale-job skip and the worker body
    never runs. That skip is CORRECT production behavior (no LLM call is burnt),
    but it is unrelated to what these tests cover, and it is exactly what CI hit:
    each test file is its own fresh pytest process (tests.yml:116-140) and the
    target test is this file's first compression test, so it paid the import
    inside its own 2.0s ceiling.

    The prime is one real, instant attempt through the production entry point
    with a generous ceiling. It warms the lazy imports and spawns the shared
    compress pool's worker thread, and it touches none of the events or fences
    the tests below assert on.
    """
    messages = [{"role": "user", "content": "keep"}]

    def instant_worker(fence: CompressionCommitFence):
        fence.touch_progress()
        return (messages, "prime")

    run_compress_context_with_progress_timeout(
        worker=instant_worker,
        messages=messages,
        system_prompt_fallback="fallback",
        idle_timeout_seconds=30.0,
        total_ceiling_seconds=60.0,
        stall_fallback=False,
    )


@contextmanager
def _prompt_start_pool():
    """Patch the compress pool so the submitted job starts immediately."""
    executor = _PromptStartExecutor()
    with patch(
        "agent.conversation_compression._get_compress_timeout_executor",
        lambda: executor,
    ):
        try:
            yield executor
        finally:
            executor.shutdown()


@contextmanager
def _compression_logs(level: int = logging.WARNING):
    """Collect records emitted by the compression module logger."""
    records: list[str] = []

    class _Capture(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    logger = logging.getLogger("agent.conversation_compression")
    handler = _Capture(level=level)
    logger.addHandler(handler)
    try:
        yield records
    finally:
        logger.removeHandler(handler)


def _build_agent(tmp_path: Path, session_id: str, db: SessionDB | None = None):
    if db is None:
        db = SessionDB(db_path=tmp_path / "state.db")
        db.create_session(session_id, source="cli")
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            quiet_mode=True,
            session_db=db,
            session_id=session_id,
            skip_context_files=True,
            skip_memory=True,
        )
    agent._compression_feasibility_checked = True
    agent.compression_in_place = True
    agent._cached_system_prompt = "sys"
    agent.context_compressor.threshold_tokens = 1_000
    return db, agent


def _messages():
    return [{"role": "user", "content": f"m{i}"} for i in range(20)]


class TestWorkerTeardownOnCeiling:
    def test_grace_env_override_controls_join_not_ceiling_cap(self):
        from agent.conversation_compression import _cancelled_worker_teardown_grace_seconds

        assert _cancelled_worker_teardown_grace_seconds(0.2) == 0.2
        with patch.dict(
            os.environ,
            {"HERMES_COMPRESSION_CANCELLED_WORKER_GRACE_SECONDS": "2.5"},
        ):
            assert _cancelled_worker_teardown_grace_seconds(0.2) == 2.5

    def test_cooperative_worker_joined_within_grace(self):
        """A worker that exits promptly after cancel is joined on the
        total-ceiling path; the lease is released normally (no retention) —
        the sabotage check for this test is removing the
        `_join_cancelled_worker` call, which makes
        `worker_done.is_set()` False when the host returns."""
        original = [{"role": "user", "content": "keep"}]
        worker_done = threading.Event()
        worker_started = threading.Event()

        def cooperative_worker(fence: CompressionCommitFence):
            # Continuous progress (the #97488 'last progress 0.0s ago'
            # shape) so only the TOTAL ceiling expires; poll the poison
            # fence like the production worker does between provider phases.
            worker_started.set()
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                if fence.is_cancelled:
                    break
                fence.touch_progress()
                time.sleep(0.01)
            # Cooperative-but-not-instant exit: the unwind after seeing the
            # poison takes real time (rollback, telemetry). It must be long
            # enough that NO host return path can beat it WITHOUT the
            # bounded-grace join — the old 0.08s raced the host's own return
            # (the host returns ~40ms after the ceiling expires), so the
            # sabotage check below passed ~1 run in 6 with the join removed.
            # 0.5s is >10x that race margin and still ~10x inside the 5s
            # grace, so it cannot make the with-join case slow or flaky.
            time.sleep(0.5)
            worker_done.set()
            return (original, "late")

        # Only the TOTAL ceiling may end this attempt, so ceiling/idle stays
        # <= 5 (the ceiling-tight guard in the wait loop). The fence is created
        # INSIDE the warmed context so its budget covers the attempt only, never
        # the machinery that has to exist before the attempt can start.
        with (
            patch.dict(
                os.environ,
                {"HERMES_COMPRESSION_CANCELLED_WORKER_GRACE_SECONDS": "5"},
            ),
            # Guarantee the job reaches its first bytecode before the host
            # starts its ceiling clock. Without this the shared daemon pool can
            # lose the start race under load, the F6 stale-job skip fires, and
            # the test asserts on a worker that never ran.
            _prompt_start_pool(),
        ):
            fence = CompressionCommitFence()
            msgs, prompt = run_compress_context_with_progress_timeout(
                worker=cooperative_worker,
                messages=original,
                system_prompt_fallback="fallback",
                idle_timeout_seconds=0.5,
                total_ceiling_seconds=2.0,
                fence=fence,
                stall_fallback=False,
            )
        # PRECONDITION, not a flake symptom: this test exercises the
        # TOTAL-CEILING path, and that path is the only one that joins
        # (the idle-stall path deliberately skips the join —
        # conversation_compression.py:1787-1792). Assert the worker actually
        # ran and the attempt ended on the ceiling, so a start-race or a
        # path mix-up reports itself instead of masquerading as a missing
        # join. On a loaded box the pool thread can miss the window entirely
        # and ``_fence_gated_worker`` skips the stale job (no LLM call burnt);
        # that is correct production behavior and not what this test claims to
        # cover, so it must fail loudly rather than intermittently.
        assert worker_started.is_set(), (
            "the compression worker never started — the F6 stale-job skip "
            "fired before the job ran, so this test did not exercise the "
            "total-ceiling join at all (#97488)"
        )
        assert fence.progress_observed, (
            "no summary progress was reported, so the attempt could not have "
            "ended on the total ceiling (#97488)"
        )
        # The bounded-grace join must have reaped the cooperative worker
        # BEFORE the host returned.
        assert worker_done.is_set(), (
            "host returned before tearing down a cooperative cancelled "
            "worker — bounded-grace join missing (#97488)"
        )
        # Whichever return path won the race (fallback via join, or the
        # worker's own return adopted inside the final wait slice), the
        # transcript must be unchanged.
        assert msgs == [{"role": "user", "content": "keep"}]
        assert prompt in ("fallback", "late")
        # Teardown proved quiescence, so the lease must NOT stay retained.
        assert fence._retain_cancelled_lock_until_worker_done is False

    def test_cooperative_worker_joined_when_scheduler_blips_before_ceiling(self):
        """Scheduler starvation can look like idle stall while the total
        ceiling is still binding (#97488). The host must wait for the ceiling,
        join on the total-ceiling path, and not return before the worker
        unwinds."""
        original = [{"role": "user", "content": "keep"}]
        worker_done = threading.Event()

        def cooperative_worker(fence: CompressionCommitFence):
            fence.touch_progress()
            # Scheduler starvation: no progress touch until after the idle
            # budget would have fired if the host treated this as idle-stall.
            time.sleep(0.6)
            while True:
                if fence.is_cancelled:
                    break
                fence.touch_progress()
                time.sleep(0.01)
            # Same widened unwind as the sibling test: must exceed any host
            # return path so the assertion has teeth without the join.
            time.sleep(0.5)
            worker_done.set()
            return (original, "late")

        # ceiling/idle <= 5 keeps the ceiling-tight guard armed (so the false
        # idle below cannot take the idle-stall path).
        with (
            patch.dict(
                os.environ,
                {"HERMES_COMPRESSION_CANCELLED_WORKER_GRACE_SECONDS": "5"},
            ),
            _prompt_start_pool(),
        ):
            # Armed after the machinery is warm: this fence's budget must cover
            # the attempt, not the cost of making the attempt possible.
            fence = CompressionCommitFence()
            msgs, prompt = run_compress_context_with_progress_timeout(
                worker=cooperative_worker,
                messages=original,
                system_prompt_fallback="fallback",
                idle_timeout_seconds=0.5,
                total_ceiling_seconds=2.0,
                fence=fence,
                stall_fallback=False,
            )
        assert worker_done.is_set(), (
            "ceiling-bound false-idle must still join cooperative workers "
            "(#97488 / #76354 S3)"
        )
        assert msgs == [{"role": "user", "content": "keep"}]
        assert prompt in ("fallback", "late")

    def test_uninterruptible_worker_is_orphaned_with_lease_retained(self):
        """A worker stuck in an uninterruptible provider call is orphaned:
        the host returns after the grace, the poison fence discards its late
        result, and on the total-ceiling path the durable lease release hook
        does NOT fire while the worker is alive (no overlap window)."""
        original = [{"role": "user", "content": "keep"}]
        release = threading.Event()
        worker_finished = threading.Event()
        lock_released: list[float] = []

        def stuck_worker(fence: CompressionCommitFence):
            # Continuous progress so only the TOTAL ceiling can expire
            # (the #97488 'last progress 0.0s ago' shape).
            while not release.wait(timeout=0.02):
                fence.touch_progress()
            worker_finished.set()
            if not fence.begin_commit():
                return (original, "")
            try:
                return ([{"role": "assistant", "content": "late"}], "late")
            finally:
                fence.finish_commit()

        fence = CompressionCommitFence()
        fence.register_cancelled_lock_release(
            lambda: lock_released.append(time.monotonic())
        )
        # ceiling/idle <= 5 so only the total ceiling can expire, with enough
        # budget for the pool thread to start the (wedged) worker first.
        # The explicit grace keeps the orphan wait short now that the ceiling
        # is generous — it only has to prove the worker does NOT exit.
        with patch.dict(
            os.environ,
            {"HERMES_COMPRESSION_CANCELLED_WORKER_GRACE_SECONDS": "0.5"},
        ):
            msgs, prompt = run_compress_context_with_progress_timeout(
                worker=stuck_worker,
                messages=original,
                system_prompt_fallback="fallback",
                idle_timeout_seconds=0.5,
                total_ceiling_seconds=2.0,
                fence=fence,
                stall_fallback=False,
            )
        # Precondition: the worker is genuinely still running.
        assert not worker_finished.is_set()
        assert msgs is original and prompt == "fallback"
        # Total-ceiling path: lease retained until the worker exits, so no
        # new attempt can overlap the unchanged session.
        assert not lock_released, (
            "durable lease released while the timed-out worker was still "
            "alive — overlap window reopened (#97488)"
        )
        release.set()
        assert worker_finished.wait(timeout=2)
        # Late result was fence-poisoned, never adopted.
        assert msgs == [{"role": "user", "content": "keep"}]

    def test_join_reads_every_settled_worker_as_exited(self):
        """Every SETTLED future means the worker thread is gone and holds no
        provider call. ``concurrent.futures.TimeoutError`` is the builtin
        ``TimeoutError`` on 3.11+, so ``Future.result(timeout=...)`` re-raises
        a worker-raised TimeoutError (the #76354 F6 pre-start stale-job skip)
        instead of reporting a grace expiry — only a still-pending future is an
        orphan. Mutation check: return False for a settled TimeoutError."""
        from agent.conversation_compression import _join_cancelled_worker

        def settled(*, result=None, exc=None) -> bool:
            future = concurrent.futures.Future()
            if exc is None:
                future.set_result(result)
            else:
                future.set_exception(exc)
            return _join_cancelled_worker(future, 5.0)

        assert settled(result=([], "p")) is True
        assert settled(exc=ValueError("worker blew up")) is True
        assert settled(exc=concurrent.futures.TimeoutError("deadline expired")) is True

        never_started = concurrent.futures.Future()
        never_started.cancel()
        assert _join_cancelled_worker(never_started, 5.0) is True

        still_running = concurrent.futures.Future()
        assert _join_cancelled_worker(still_running, 0.05) is False, (
            "a worker that has NOT exited must still be reported as an orphan "
            "so its durable lease stays retained (#97488)"
        )

    def test_late_started_worker_is_joined_not_reported_orphaned(self):
        """A pool thread that starts the job after the ceiling makes
        ``_fence_gated_worker`` skip the summary as stale: the worker exits by
        raising TimeoutError having never run a provider call. The bounded
        grace join must read that settled exit as "exited" — no phantom
        orphan warning, and the total-ceiling durable lease released rather
        than retained (#97488). Mutation check: the un-guarded
        ``except TimeoutError: return False``."""
        original = [{"role": "user", "content": "keep"}]
        worker_ran = threading.Event()

        def stale_worker(fence: CompressionCommitFence):
            worker_ran.set()
            fence.touch_progress()
            while not fence.is_cancelled:
                time.sleep(0.01)
            return (original, "late")

        fence = CompressionCommitFence()
        with (
            patch(
                "agent.conversation_compression._get_compress_timeout_executor",
                lambda: _LateStartExecutor(delay=0.6),
            ),
            patch.dict(
                os.environ,
                {"HERMES_COMPRESSION_CANCELLED_WORKER_GRACE_SECONDS": "5"},
            ),
            _compression_logs() as records,
        ):
            msgs, prompt = run_compress_context_with_progress_timeout(
                worker=stale_worker,
                messages=original,
                system_prompt_fallback="fallback",
                idle_timeout_seconds=0.1,
                total_ceiling_seconds=0.2,
                fence=fence,
                stall_fallback=False,
            )

        # Precondition: the stale-job skip meant the worker never ran.
        assert not worker_ran.is_set()
        assert msgs is original and prompt == "fallback"
        assert not any("did not exit within" in rec for rec in records), records
        assert fence._retain_cancelled_lock_until_worker_done is False, (
            "durable lease retained for a worker that provably exited — the "
            "session's next compression attempt would be locked out (#97488)"
        )

    def test_worker_timeout_exit_does_not_spin_until_the_ceiling(self):
        """A worker that exits by raising TimeoutError ends the attempt, so the
        host must stop there instead of re-polling the settled future in 5ms
        slices (CPU-burning spin plus an INFO "still streaming" flood) until
        the ceiling expires."""
        def timing_out_worker(fence: CompressionCommitFence):
            raise concurrent.futures.TimeoutError("provider call timed out")

        fence = CompressionCommitFence()
        started = time.monotonic()
        with (
            _compression_logs(level=logging.INFO) as records,
            patch.dict(
                os.environ,
                {"HERMES_COMPRESSION_CANCELLED_WORKER_GRACE_SECONDS": "5"},
            ),
        ):
            msgs, prompt = run_compress_context_with_progress_timeout(
                worker=timing_out_worker,
                messages=[{"role": "user", "content": "keep"}],
                system_prompt_fallback="fallback",
                idle_timeout_seconds=2.0,
                total_ceiling_seconds=4.0,
                fence=fence,
                stall_fallback=False,
            )
        elapsed = time.monotonic() - started

        assert msgs == [{"role": "user", "content": "keep"}]
        assert prompt == "fallback"
        assert elapsed < 2.0, (
            f"host re-polled a settled future for {elapsed:.1f}s instead of "
            f"returning when the worker exited"
        )
        assert not any("still streaming" in rec for rec in records), (
            "settled worker re-polled until the ceiling (INFO flood)"
        )

    def test_worker_timeout_exit_mid_commit_stops_the_overrun_loop(self):
        """A worker that raises TimeoutError inside its commit has unwound:
        no commit is in flight, so the host must stop re-raising the settled
        exception and stop re-reporting the overrun forever."""
        original = [{"role": "user", "content": "keep"}]
        entered = threading.Event()
        overruns: list[float] = []

        def timing_out_commit_worker(fence: CompressionCommitFence):
            assert fence.begin_commit()
            entered.set()
            try:
                # Outlast the total ceiling, then exit by raising TimeoutError
                # while holding the commit fence.
                time.sleep(0.2)
                raise concurrent.futures.TimeoutError("SessionDB write timed out")
            finally:
                fence.finish_commit()

        def on_commit_overrun(waited: float, _ceiling: float) -> None:
            # NOTE: the host deliberately swallows exceptions from this
            # callback, so the teeth here are the host's own return (a
            # re-polled settled future never returns), not the count.
            overruns.append(waited)

        fence = CompressionCommitFence()
        result: dict = {}

        def run():
            result["value"] = run_compress_context_with_progress_timeout(
                worker=timing_out_commit_worker,
                messages=original,
                system_prompt_fallback="fallback",
                idle_timeout_seconds=0.05,
                total_ceiling_seconds=0.05,
                on_commit_overrun=on_commit_overrun,
                fence=fence,
                stall_fallback=False,
            )

        t = threading.Thread(target=run, daemon=True)
        t.start()
        assert entered.wait(timeout=2)
        t.join(timeout=5)

        assert not t.is_alive(), (
            "host must return once the worker's commit has unwound"
        )
        assert result.get("value") == (original, "fallback")
        assert len(overruns) <= 1
        assert fence._retain_cancelled_lock_until_worker_done is False


class TestDurableAttemptBackoff:
    def test_backoff_row_records_strategy_and_kind(self, tmp_path: Path):
        db, agent = _build_agent(tmp_path, "BACKOFF_KIND")
        agent.context_compressor.record_timeout_failure(
            "host ceiling exhausted", failure_kind="ceiling_exhausted"
        )
        row = db.get_compression_failure_cooldown("BACKOFF_KIND")
        assert row is not None, "backoff must persist to state.db"
        assert row["remaining_seconds"] > 0
        assert "backoff:ceiling_exhausted:strategy=lean" in (row["error"] or "")

    def test_backoff_blocks_same_strategy_reentry_next_turn(self, tmp_path: Path):
        db, agent = _build_agent(tmp_path, "BACKOFF_REENTRY")
        agent.context_compressor.record_timeout_failure(
            "stall", failure_kind="stalled"
        )
        # Precondition: over threshold, so only the backoff can block.
        assert 500_000 >= agent.context_compressor.threshold_tokens
        should, reason = agent.context_compressor.should_compress_info(500_000)
        assert should is False
        assert reason and reason.startswith("cooldown"), (
            "next-turn re-entry of the same strategy must be skipped inside "
            "the backoff window (#96775)"
        )

    def test_backoff_survives_simulated_gateway_restart(self, tmp_path: Path):
        db, agent = _build_agent(tmp_path, "BACKOFF_RESTART")
        agent.context_compressor.record_timeout_failure(
            "stall before restart", failure_kind="stall_interrupted"
        )
        # Precondition: row is durable in this DB file.
        assert db.get_compression_failure_cooldown("BACKOFF_RESTART")
        # Simulated restart: brand-new SessionDB handle + brand-new agent
        # objects rebuilt from the same state.db file.
        db2 = SessionDB(db_path=tmp_path / "state.db")
        _db2, agent2 = _build_agent(tmp_path, "BACKOFF_RESTART", db=db2)
        cooldown = agent2.context_compressor.get_active_compression_failure_cooldown(
            refresh=True
        )
        assert cooldown is not None, (
            "backoff must survive a gateway restart via state.db (#96775)"
        )
        assert "stall_interrupted" in (cooldown["error"] or "")
        should, reason = agent2.context_compressor.should_compress_info(500_000)
        assert should is False and reason.startswith("cooldown")

    def test_success_clears_backoff(self, tmp_path: Path):
        db, agent = _build_agent(tmp_path, "BACKOFF_CLEAR")
        compressor = agent.context_compressor
        compressor.record_timeout_failure("stall", failure_kind="stalled")
        assert db.get_compression_failure_cooldown("BACKOFF_CLEAR")
        # What a successful compression does on commit:
        compressor._clear_compression_failure_cooldown()
        assert db.get_compression_failure_cooldown("BACKOFF_CLEAR") is None
        assert compressor.should_compress_info(500_000)[0] is True


class TestSupersessionDiscardsLateResults:
    def test_superseded_attempt_candidate_never_commits(self, tmp_path: Path):
        db, agent = _build_agent(tmp_path, "SUPERSEDE")
        live = _messages()
        original = copy.deepcopy(live)

        def compress_and_get_superseded(messages, **_kwargs):
            # While this attempt's summary was in flight, a NEWER attempt
            # claimed the compressor (what a retry/fallback does).
            _claim_compressor_attempt(agent.context_compressor)
            return [{"role": "assistant", "content": "stale summary"}]

        agent.context_compressor.compress = compress_and_get_superseded
        out, _prompt = compress_context(
            agent, live, "sys", approx_tokens=500_000
        )
        assert out == original, (
            "late candidate from a superseded attempt must be discarded, "
            "never committed over newer state (#97488)"
        )
        assert live == original
        # Session stayed writable and unrotated.
        assert db.get_compression_lock_holder("SUPERSEDE") is None
        db.append_message("SUPERSEDE", "assistant", "still writable")


class TestTransientBlockIsNotExhaustion:
    def test_cooldown_blocked_noop_sets_transient_signal(self, tmp_path: Path):
        db, agent = _build_agent(tmp_path, "TRANSIENT_SIGNAL")
        agent.context_compressor.record_timeout_failure(
            "host ceiling", failure_kind="ceiling_exhausted"
        )
        live = _messages()
        before = copy.deepcopy(live)
        out, _ = compress_context(agent, live, "sys", approx_tokens=500_000)
        # Preconditions: the pass no-oped and it was NOT a lock skip.
        assert out == before
        assert getattr(agent, "_compression_skipped_due_to_lock", None) is None
        assert compression_blocked_transiently(agent) is True, (
            "a cooldown-blocked no-op must be distinguishable from "
            "exhaustion or the gateway falsely auto-resets (#97488)"
        )

    def test_signal_cleared_per_attempt_and_not_set_when_unblocked(
        self, tmp_path: Path
    ):
        db, agent = _build_agent(tmp_path, "TRANSIENT_CLEAR")
        # Stale signal from a previous pass must not leak.
        agent._compression_blocked_transient = "cooldown:999"
        agent.context_compressor.compress = lambda messages, **kw: list(messages)
        live = _messages()
        compress_context(agent, live, "sys", approx_tokens=500_000)
        assert compression_blocked_transiently(agent) is False

    def test_type_pinned_against_magicmock_agents(self):
        from unittest.mock import MagicMock

        mock_agent = MagicMock()
        # MagicMock auto-attributes are truthy but not str.
        assert compression_blocked_transiently(mock_agent) is False


def _summary_response(content: str):
    from unittest.mock import MagicMock

    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = content
    return response


class TestProviderOverflowBypassesCooldown:
    """#100661: a provider-proven overflow must get one REAL summary attempt
    while the summary-failure cooldown is armed. Before the fix every turn of
    a wedged session hit the cooldown gate, returned the soft "temporarily
    paused" deferral, and the next failure extended the ladder — 4 long
    sessions were lost this way. Ordinary (non-overflow) automatic passes
    must still defer."""

    def _armed_agent(self, tmp_path: Path, session_id: str):
        db, agent = _build_agent(tmp_path, session_id)
        # Realistic arming: a failed/stalled attempt recorded the ladder.
        agent.context_compressor.record_timeout_failure(
            "stall", failure_kind="stalled"
        )
        assert agent.context_compressor.should_compress_info(500_000)[0] is False
        return db, agent

    def test_overflow_attempt_invokes_summarizer_while_cooldown_armed(
        self, tmp_path: Path
    ):
        db, agent = self._armed_agent(tmp_path, "OVERFLOW_BYPASS")
        calls = []

        def fake_call_llm(**kwargs):
            calls.append(kwargs)
            return _summary_response("## Goal\nRecovered after overflow.")

        # Bulky turns so the compacted transcript is genuinely smaller.
        live = [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i} " * 400}
            for i in range(20)
        ]
        with patch("agent.context_compressor.call_llm", fake_call_llm):
            out, _ = compress_context(
                agent, live, "sys", approx_tokens=500_000, bypass_cooldown=True
            )
        assert len(calls) == 1, (
            "provider-proven overflow must reach the summary LLM even while "
            "the failure cooldown is armed (#100661)"
        )
        assert compression_blocked_transiently(agent) is False
        assert len(out) < len(live), "the attempt must actually compact"

    def test_non_overflow_pass_still_deferred_by_cooldown(self, tmp_path: Path):
        db, agent = self._armed_agent(tmp_path, "OVERFLOW_ORDINARY")
        calls = []

        def fake_call_llm(**kwargs):  # pragma: no cover - must not run
            calls.append(kwargs)
            return _summary_response("unexpected")

        live = _messages()
        before = copy.deepcopy(live)
        with patch("agent.context_compressor.call_llm", fake_call_llm):
            out, _ = compress_context(agent, live, "sys", approx_tokens=500_000)
        assert calls == [] and out == before
        assert compression_blocked_transiently(agent) is True, (
            "ordinary threshold pressure keeps honoring the cooldown (#11529)"
        )
