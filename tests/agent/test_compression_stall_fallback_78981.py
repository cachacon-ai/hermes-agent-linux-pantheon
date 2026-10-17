"""Compression falls back after an aborted (stalled) summary — #78981.

A summariser that keeps the connection open but never emits a real token
produces no fence progress, so the host's progress-aware timeout aborts the
worker and returns "continue without compression". Nothing raises out of the
auxiliary client on that path, so its configured ``fallback_chain`` — the
user's declared answer to "this route is unhealthy" — was never consulted for
the one failure mode that most needs it.

These tests pin the contract:

* an aborted stall re-attempts compression once with the summary route pinned
  to the configured ``auxiliary.compression.fallback_chain``;
* the pinned route reaches the summary ``call_llm`` (provider/model/base_url/
  api_key/timeout), and is single-use so the compressor's own main-model retry
  does not re-issue the same failed route;
* the historical "continue without compression" degrade survives when no chain
  is configured or the fallback attempt also stalls.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace
from unittest.mock import patch

from agent.context_compressor import (
    ContextCompressor,
    pin_summary_route,
    take_pinned_summary_route,
)
from agent.conversation_compression import (
    CompressionCommitFence,
    resolve_compression_fallback_route,
    run_compress_context_with_progress_timeout,
)

CHAIN_ENTRY = {
    "provider": "custom",
    "model": "backup-summarizer",
    "base_url": "https://fallback.invalid/v1",
    "api_key": "sk-fallback",
    "timeout": 45,
}


def _patch_chain(chain):
    """Pin auxiliary.compression config without touching the real config.yaml."""
    return patch(
        "agent.auxiliary_client._get_auxiliary_task_config",
        return_value={"fallback_chain": chain},
    )


class _StalledSummaryWorker:
    """A compression worker whose unpinned (primary) attempt streams nothing.

    Mirrors the reported shape: the provider holds the connection open, so the
    worker never calls ``fence.touch_progress()`` and the host's idle budget
    lapses.

    The stall keys on the attempt's ROUTE, never on how many times the worker has
    been called. When the host is starved enough to poison the fence before the
    pool thread starts, ``_fence_gated_worker`` skips the primary job entirely
    so a stale job never burns an LLM call (#76354 F6) — and the retry's pinned
    invocation is then the FIRST one this worker ever sees. Counting
    invocations there stalls the very attempt the contract is about and makes
    "exactly two invocations" unreachable, even though production is behaving
    correctly.

    ``stall_pinned`` additionally hangs the pinned fallback, for the "this route
    is unhealthy too" case, and ``stalled_abort`` is what that hung attempt
    hands back once the stall is released: the caller's own message list plus
    the degraded prompt — the "aborted, nothing compressed" result production
    sees from a provider call its fence has already cut off. Handing back a real
    compression instead would publish a fallback that never actually ran.
    """

    def __init__(self, compressed, *, stall_pinned=False, stalled_abort=None):
        self.compressed = compressed
        self.stall_pinned = stall_pinned
        self.stalled_abort = stalled_abort
        self.routes = []
        self.fences = []
        self._lock = threading.Lock()
        self.release = threading.Event()
        # Set the moment a pinned attempt is really invoked. Proves the attempt
        # STARTED, which a timeout alone cannot: starvation can end the host's
        # wait on an attempt that never ran.
        self.pinned_attempt_entered = threading.Event()

    @property
    def pinned(self):
        """Attempts that ran with a pinned fallback route, in order."""
        return [route for route in self.routes if route is not None]

    def __call__(self, fence: CompressionCommitFence):
        with self._lock:
            route = take_pinned_summary_route()
            self.routes.append(route)
            self.fences.append(fence)
        if route is None or self.stall_pinned:
            if route is not None:
                self.pinned_attempt_entered.set()
            # Connection open, zero tokens, zero fence progress.
            self.release.wait(timeout=10)
            if route is not None and self.stalled_abort is not None:
                return self.stalled_abort
            return ([{"role": "assistant", "content": "late"}], "late-prompt")
        if not fence.begin_commit():
            return ([{"role": "assistant", "content": "cancelled"}], "cancelled")
        try:
            return (self.compressed, "summarized-prompt")
        finally:
            fence.finish_commit()


def _run(worker, *, chain, timeouts, messages, idle=0.05, ceiling=0.2, fence=None):
    with _patch_chain(chain):
        return run_compress_context_with_progress_timeout(
            worker=worker,
            messages=messages,
            system_prompt_fallback="degraded-prompt",
            idle_timeout_seconds=idle,
            total_ceiling_seconds=ceiling,
            on_timeout=lambda *args: timeouts.append(args),
            fence=fence,
        )


# ---------------------------------------------------------------------------
# Fence-level contract: an aborted stall consults the configured chain
# ---------------------------------------------------------------------------


def test_stalled_summary_attempts_configured_fallback_chain():
    original = [{"role": "user", "content": "keep-me"}]
    compressed = [{"role": "user", "content": "summary of earlier turns"}]
    worker = _StalledSummaryWorker(compressed)
    timeouts = []
    # Host-owned fence, exactly as run_agent.py hands its active one in: the
    # test keeps the object so the primary attempt's state stays observable even
    # when #76354 F6 skips that attempt before it ever runs.
    primary_fence = CompressionCommitFence()

    try:
        msgs, prompt = _run(
            worker,
            chain=[CHAIN_ENTRY],
            timeouts=timeouts,
            messages=original,
            fence=primary_fence,
        )
    finally:
        worker.release.set()

    assert len(worker.pinned) == 1, "the aborted stall must be retried exactly once"
    # Only meaningful when the primary actually ran — under starvation the
    # pinned retry can be this worker's FIRST and only invocation.
    if worker.routes[:-1]:
        assert all(route is None for route in worker.routes[:-1]), (
            "the primary attempt is never pinned"
        )
    assert primary_fence.is_cancelled, (
        "the aborted attempt stays cancelled — a late primary commit cannot land"
    )
    pinned = worker.pinned[0]
    assert pinned["provider"] == "custom"
    assert pinned["model"] == "backup-summarizer"
    assert msgs == compressed, "the fallback attempt's compression must be published"
    assert prompt == "summarized-prompt"
    assert not timeouts, "no continue-without-compression degrade after a recovery"


def test_retry_runs_on_a_host_published_fence():
    """The aborted fence vetoes every future commit, so the retry needs a new
    one — minted through the host so ``/stop`` admits against the attempt that
    is actually running."""
    original = [{"role": "user", "content": "keep-me"}]
    compressed = [{"role": "user", "content": "summary"}]
    worker = _StalledSummaryWorker(compressed)
    minted = []
    primary_fence = CompressionCommitFence()

    def _new_fence():
        fence = CompressionCommitFence()
        minted.append(fence)
        return fence

    try:
        with _patch_chain([CHAIN_ENTRY]):
            msgs, _prompt = run_compress_context_with_progress_timeout(
                worker=worker,
                messages=original,
                system_prompt_fallback="degraded-prompt",
                idle_timeout_seconds=0.05,
                total_ceiling_seconds=0.2,
                fence=primary_fence,
                new_fence=_new_fence,
            )
    finally:
        worker.release.set()

    assert msgs == compressed
    assert len(minted) == 1, "exactly one fence is minted for the one retry"
    assert worker.fences[-1] is minted[0], "the retry runs on the minted fence"
    aborted_fences = worker.fences[:-1]
    # Vacuous when the primary was skipped before it ran (#76354 F6), so the
    # unconditional fence-state assertion below carries the check there.
    if aborted_fences:
        assert all(f is not minted[0] for f in aborted_fences), (
            "the retry must not reuse the aborted attempt's fence"
        )
        assert all(f.is_cancelled for f in aborted_fences), (
            "the aborted attempt stays cancelled"
        )
    assert primary_fence.is_cancelled, (
        "the aborted attempt stays cancelled — a late primary commit cannot land"
    )


def test_hard_interrupt_suppresses_the_fallback_attempt():
    """An explicit stop is not an unhealthy route — don't start another
    summary on the user's behalf after they asked for the turn to end."""
    original = [{"role": "user", "content": "keep-me"}]
    worker = _StalledSummaryWorker([{"role": "user", "content": "unused"}])
    stopped = threading.Event()
    stopped.set()
    agent = SimpleNamespace(_hard_interrupt_requested=stopped)
    timeouts = []

    try:
        with _patch_chain([CHAIN_ENTRY]):
            msgs, prompt = run_compress_context_with_progress_timeout(
                worker=worker,
                messages=original,
                system_prompt_fallback="degraded-prompt",
                idle_timeout_seconds=0.05,
                total_ceiling_seconds=0.2,
                on_timeout=lambda *args: timeouts.append(args),
                telemetry_agent=agent,
            )
    finally:
        worker.release.set()

    assert not worker.pinned, "an explicit stop must not start a fallback summary"
    assert msgs is original
    assert prompt == "degraded-prompt"
    assert len(timeouts) == 1


def test_no_fallback_chain_configured_degrades_without_retry():
    original = [{"role": "user", "content": "keep-me"}]
    worker = _StalledSummaryWorker([{"role": "user", "content": "unused"}])
    timeouts = []

    try:
        msgs, prompt = _run(worker, chain=[], timeouts=timeouts, messages=original)
    finally:
        worker.release.set()

    assert not worker.pinned, "nothing to fall back to — do not burn a retry"
    assert msgs is original
    assert prompt == "degraded-prompt"
    assert len(timeouts) == 1


def test_fallback_that_also_stalls_degrades_after_one_attempt():
    """The pinned retry stalls too — so the run degrades, after ONE attempt.

    The retry inherits the chain entry's own 45s budget (production semantics,
    #62452), which is what makes the attempt's START deterministic: no amount
    of scheduler starvation can expire the retry's idle window before the pool
    thread has invoked the worker, so this test cannot fail as ``0 == 1`` with
    the retry silently skipped — the same flake the primary hop had. The stall
    is released through an explicit handshake rather than by waiting the
    declared timeout out, so the test stays fast.

    The host runs on its own thread because the handshake and the stall have to
    be observed from outside the wait it is blocking.
    """
    original = [{"role": "user", "content": "keep-me"}]
    worker = _StalledSummaryWorker(
        [{"role": "user", "content": "unused"}],
        stall_pinned=True,
        stalled_abort=(original, "degraded-prompt"),
    )
    timeouts = []
    outcome = {}
    host_done = threading.Event()

    def _host():
        try:
            outcome["result"] = _run(
                worker, chain=[CHAIN_ENTRY], timeouts=timeouts, messages=original
            )
        except BaseException as exc:  # surfaced by the caller below
            outcome["error"] = exc
        finally:
            host_done.set()

    host = threading.Thread(target=_host, name="stall-fallback-host", daemon=True)
    host.start()
    try:
        entered = worker.pinned_attempt_entered.wait(timeout=10)
        # Observed while the double is still parked in its stall.
        parked = not host_done.is_set()
        pinned_seen = len(worker.pinned)
    finally:
        worker.release.set()
        finished = host_done.wait(timeout=15)

    assert entered, "the pinned retry must actually invoke the worker"
    assert parked, (
        "the stalled fallback must hold the host — a retry that returned "
        "without compressing is what degrades, not a retry that ran"
    )
    assert pinned_seen == 1, "the fallback is attempted once, not in a loop"
    assert finished, "the host must return once the stalled fallback unwinds"
    if "error" in outcome:
        raise outcome["error"]
    msgs, prompt = outcome["result"]
    assert msgs is original, "no messages may be dropped when both routes stall"
    assert prompt == "degraded-prompt"
    assert len(timeouts) == 1, "the degrade must be reported exactly once"


# ---------------------------------------------------------------------------
# Route resolution: a chain entry becomes an explicit summary route
# ---------------------------------------------------------------------------


def test_resolved_route_carries_entry_credentials_and_timeout():
    with _patch_chain([CHAIN_ENTRY]):
        route = resolve_compression_fallback_route()

    assert route is not None
    assert route["provider"] == "custom"
    assert route["model"] == "backup-summarizer"
    assert route["base_url"] == "https://fallback.invalid/v1"
    assert route["api_key"] == "sk-fallback"
    # Per-entry timeouts already govern aux-client fallback candidates
    # (#62452); the stall retry honours the same declaration.
    assert route["timeout"] == 45.0


def test_incomplete_chain_entries_are_skipped():
    chain = [
        "not-a-mapping",
        {"model": "orphan-model"},          # no provider
        {"provider": "custom"},             # no model
        CHAIN_ENTRY,
    ]
    with _patch_chain(chain):
        route = resolve_compression_fallback_route()

    assert route is not None
    assert route["model"] == "backup-summarizer"


def test_no_chain_resolves_to_no_route():
    with _patch_chain([]):
        assert resolve_compression_fallback_route() is None


# ---------------------------------------------------------------------------
# Injection point: the pinned route reaches the summary call
# ---------------------------------------------------------------------------


def _make_compressor(summary_model="aux-summarizer"):
    with patch(
        "agent.context_compressor.get_model_context_length", return_value=100000
    ):
        return ContextCompressor(
            model="main-model",
            quiet_mode=True,
            summary_model_override=summary_model,
        )


def _msgs():
    return [
        {"role": "user", "content": "u1 " + "x" * 200},
        {"role": "assistant", "content": "a1 " + "y" * 200},
        {"role": "user", "content": "u2 " + "z" * 200},
    ]


def _ok_response(content="SUMMARY BODY"):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


def test_pinned_route_overrides_the_summary_call_route():
    compressor = _make_compressor()
    calls = []

    def _fake_call_llm(**kwargs):
        calls.append(kwargs)
        return _ok_response()

    with patch("agent.context_compressor.call_llm", side_effect=_fake_call_llm):
        with pin_summary_route(dict(CHAIN_ENTRY)):
            summary = compressor._generate_summary(_msgs())

    assert summary and "SUMMARY BODY" in summary
    assert len(calls) == 1
    call = calls[0]
    assert call["task"] == "compression"
    assert call["provider"] == "custom"
    assert call["model"] == "backup-summarizer"
    assert call["base_url"] == "https://fallback.invalid/v1"
    assert call["api_key"] == "sk-fallback"
    assert call["timeout"] == 45


def test_pinned_route_is_not_reissued_by_the_main_model_retry():
    """The compressor's own main-model retry must not re-run the failed route."""
    compressor = _make_compressor()
    calls = []

    def _fake_call_llm(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise TimeoutError("Request timed out.")
        return _ok_response()

    with patch("agent.context_compressor.call_llm", side_effect=_fake_call_llm):
        with pin_summary_route(dict(CHAIN_ENTRY)):
            summary = compressor._generate_summary(_msgs())

    assert summary and "SUMMARY BODY" in summary
    assert len(calls) == 2
    assert calls[0]["provider"] == "custom"
    assert "provider" not in calls[1], (
        "the retry must route normally, not repeat the stalled fallback route"
    )


def test_unpinned_summary_call_keeps_task_routing():
    compressor = _make_compressor()
    calls = []

    def _fake_call_llm(**kwargs):
        calls.append(kwargs)
        return _ok_response()

    with patch("agent.context_compressor.call_llm", side_effect=_fake_call_llm):
        summary = compressor._generate_summary(_msgs())

    assert summary
    assert calls and "provider" not in calls[0]
    assert calls[0]["model"] == "aux-summarizer"
