"""Host-level concurrency accounting + review-lane fairness (OOF-30 review).

Three gaps found in review of the original memory-guard PR:

1. The standalone daemon path (``hermes kanban daemon --force`` /
   :func:`hermes_cli.kanban_db.run_daemon`) never resolved
   ``kanban.max_in_progress`` at all — the one shipped entry point that
   could still fan out an entire backlog in a single tick.
2. ``max_in_progress`` was enforced per-board while the gateway dispatcher
   ticks every active board — N boards multiplied the host budget by N.
3. The ready loop consumed the entire shared spawn budget before the
   review loop ran, so a sustained ready backlog starved autonomous
   reviews indefinitely.
"""

from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _set_task_status(conn: sqlite3.Connection, task_id: str, status: str) -> None:
    conn.execute("UPDATE tasks SET status = ? WHERE id = ?", (status, task_id))


def _fake_spawn_factory(spawns: list):
    def fake_spawn(task, workspace, board=None):
        spawns.append(task.id)
        return 42
    return fake_spawn


# ---------------------------------------------------------------------------
# 1. Standalone daemon resolves max_in_progress (P1a)
# ---------------------------------------------------------------------------


def test_run_daemon_resolves_and_passes_max_in_progress(
    kanban_home, monkeypatch,
):
    """The daemon tick must pass a resolved cap into dispatch_once.

    Regression guard for the OOF-30 review finding: ``run_daemon`` only
    forwarded ``max_spawn`` — with no explicit ``--max`` (the shipped
    systemd shape) nothing capped the tick even though the gateway and
    ``hermes kanban dispatch`` paths both resolved the memory-derived
    default.
    """
    captured: dict = {}
    stop = threading.Event()

    def fake_dispatch_once(conn, **kwargs):
        captured.update(kwargs)
        return kb.DispatchResult()

    monkeypatch.setattr(kb, "dispatch_once", fake_dispatch_once)
    # No explicit config → the derived default must flow through.
    monkeypatch.setattr(kb, "configured_max_in_progress", lambda: None)
    monkeypatch.setattr(kb, "derive_default_max_in_progress", lambda sample=None: 3)

    def on_tick(res):
        stop.set()

    kb.run_daemon(interval=0.01, stop_event=stop, on_tick=on_tick)

    assert captured.get("max_in_progress") == 3


def test_run_daemon_explicit_config_wins(kanban_home, monkeypatch):
    captured: dict = {}
    stop = threading.Event()

    def fake_dispatch_once(conn, **kwargs):
        captured.update(kwargs)
        return kb.DispatchResult()

    monkeypatch.setattr(kb, "dispatch_once", fake_dispatch_once)
    monkeypatch.setattr(kb, "configured_max_in_progress", lambda: 7)
    monkeypatch.setattr(
        kb, "derive_default_max_in_progress",
        lambda sample=None: pytest.fail("derived default must not be consulted"),
    )

    def on_tick(res):
        stop.set()

    kb.run_daemon(interval=0.01, stop_event=stop, on_tick=on_tick)

    assert captured.get("max_in_progress") == 7


def test_configured_max_in_progress_parsing(monkeypatch):
    import hermes_cli.config as cfgmod

    cases = [
        ({"kanban": {"max_in_progress": 4}}, 4),
        ({"kanban": {"max_in_progress": "5"}}, 5),
        ({"kanban": {"max_in_progress": 0}}, None),
        ({"kanban": {"max_in_progress": -2}}, None),
        ({"kanban": {"max_in_progress": "lots"}}, None),
        ({"kanban": {}}, None),
        ({}, None),
    ]
    for config, expected in cases:
        monkeypatch.setattr(
            cfgmod, "load_config_readonly", lambda c=config: c
        )
        assert kb.configured_max_in_progress() == expected, config


# ---------------------------------------------------------------------------
# 2. max_in_progress counts running work on ALL boards (P1b)
# ---------------------------------------------------------------------------


def test_max_in_progress_counts_other_boards(
    kanban_home, all_assignees_spawnable,
):
    """Workers running on another board consume the same host budget."""
    kb.create_board("second")

    # Two workers already running on the second board.
    with kb.connect(board="second") as conn:
        for title in ("busy-1", "busy-2"):
            tid = kb.create_task(conn, title=title, assignee="alice")
            assert kb.claim_task(conn, tid) is not None

    spawns: list = []
    with kb.connect() as conn:
        kb.create_task(conn, title="wants-to-run", assignee="alice")
        res = kb.dispatch_once(
            conn, spawn_fn=_fake_spawn_factory(spawns), max_in_progress=2,
        )

    # Host budget (2) already consumed by the second board → nothing spawns.
    assert not spawns
    assert not res.spawned


def test_max_in_progress_partial_budget_across_boards(
    kanban_home, all_assignees_spawnable,
):
    kb.create_board("second")

    with kb.connect(board="second") as conn:
        tid = kb.create_task(conn, title="busy", assignee="alice")
        assert kb.claim_task(conn, tid) is not None

    spawns: list = []
    with kb.connect() as conn:
        for title in ("a", "b", "c"):
            kb.create_task(conn, title=title, assignee="alice")
        res = kb.dispatch_once(
            conn, spawn_fn=_fake_spawn_factory(spawns), max_in_progress=2,
        )

    # 1 running elsewhere + budget 2 → exactly one new spawn here.
    assert len(spawns) == 1
    assert len(res.spawned) == 1


def test_count_running_tasks_other_boards_fails_open(
    kanban_home, monkeypatch,
):
    """A broken board enumeration must not brick dispatch (returns 0)."""
    monkeypatch.setattr(
        kb, "list_boards",
        lambda **k: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    assert kb.count_running_tasks_other_boards() == 0


def test_concurrent_board_ticks_cannot_oversubscribe_host_cap(
    kanban_home, all_assignees_spawnable, monkeypatch,
):
    """Two boards ticking at once must never exceed ``max_in_progress``.

    Regression for the host-cap race: board-scoped dispatch locks allow
    unrelated boards to tick in parallel, and a plain read/check of
    ``count_running_tasks_other_boards`` then let each tick observe zero
    running workers and each reserve the full budget. Forcing both ticks
    through that former race window (sync inside the cross-board count)
    must still leave total running workers ≤ the host cap.
    """
    kb.create_board("second")

    with kb.connect() as conn:
        for title in ("d1", "d2", "d3"):
            kb.create_task(conn, title=title, assignee="alice")
    with kb.connect(board="second") as conn:
        for title in ("s1", "s2", "s3"):
            kb.create_task(conn, title=title, assignee="alice")

    # Hold both board locks simultaneously so the test proves the host-cap
    # lock — not the board locks — is what serializes the budget check.
    # (Board locks are independent by design; see test_lock_is_board_scoped.)
    entered_count = threading.Event()
    release_count = threading.Event()
    in_count = 0
    in_count_lock = threading.Lock()
    concurrent_count_entries = threading.Event()
    orig_count = kb.count_running_tasks_other_boards

    def racing_count(board=None):
        nonlocal in_count
        with in_count_lock:
            in_count += 1
            if in_count > 1:
                concurrent_count_entries.set()
        entered_count.set()
        # Widen the former race window: without the host-cap lock the
        # sibling tick reaches this same point and sets
        # concurrent_count_entries. With the lock it blocks before entry.
        assert release_count.wait(timeout=5), "host-cap holder stuck in count"
        with in_count_lock:
            in_count -= 1
        return orig_count(board)

    monkeypatch.setattr(kb, "count_running_tasks_other_boards", racing_count)

    results: dict[str, object] = {}
    errors: list[BaseException] = []

    def tick(slug: str) -> None:
        try:
            spawns: list = []
            with kb.connect(board=slug) as conn:
                results[slug] = kb.dispatch_once(
                    conn,
                    board=slug,
                    spawn_fn=_fake_spawn_factory(spawns),
                    max_in_progress=2,
                )
            results[f"{slug}_spawns"] = spawns
        except BaseException as exc:  # noqa: BLE001 — surface in parent
            errors.append(exc)

    t_default = threading.Thread(target=tick, args=("default",))
    t_second = threading.Thread(target=tick, args=("second",))
    t_default.start()
    t_second.start()

    assert entered_count.wait(timeout=5), "neither tick reached host-cap count"
    # Give the sibling ample time to also enter the count if the host-cap
    # lock is missing. Under the lock it must still be blocked here.
    assert not concurrent_count_entries.wait(timeout=0.3), (
        "both board ticks entered count_running_tasks_other_boards concurrently "
        "— host-cap lock is not serializing the read/check"
    )
    release_count.set()
    t_default.join(timeout=10)
    t_second.join(timeout=10)
    assert not t_default.is_alive() and not t_second.is_alive()
    assert not errors, errors

    running = {}
    for slug in ("default", "second"):
        with kb.connect(board=slug) as conn:
            running[slug] = kb.count_running_tasks(conn)
    total_running = sum(running.values())
    assert total_running <= 2, {
        "errors": errors,
        "spawned": {
            "default": len(results.get("default_spawns") or []),
            "second": len(results.get("second_spawns") or []),
        },
        "running": running,
        "total_running": total_running,
    }
    assert not concurrent_count_entries.is_set()


def test_host_cap_lock_serializes_separate_processes(kanban_home, tmp_path):
    """A second process must wait while another process holds the host lock."""
    started = tmp_path / "host-lock-child-started"
    acquired = tmp_path / "host-lock-child-acquired"
    repo_root = Path(__file__).resolve().parents[2]
    env = dict(os.environ)
    current_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(repo_root) + (
        os.pathsep + current_pythonpath if current_pythonpath else ""
    )
    child_code = (
        "from pathlib import Path\n"
        "import sys\n"
        "from hermes_cli import kanban_db as kb\n"
        "started, acquired = map(Path, sys.argv[1:])\n"
        "started.touch()\n"
        "with kb._dispatch_host_cap_lock() as locked:\n"
        "    acquired.write_text(str(locked), encoding='utf-8')\n"
    )
    child = None
    try:
        with kb._dispatch_host_cap_lock() as parent_locked:
            assert parent_locked
            child = subprocess.Popen(
                [sys.executable, "-c", child_code, str(started), str(acquired)],
                cwd=repo_root,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            deadline = time.monotonic() + 5
            while not started.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            assert started.exists(), "child process did not reach lock attempt"
            time.sleep(0.25)
            assert child.poll() is None, "child completed without waiting for the process lock"
            assert not acquired.exists(), "child bypassed the cross-process host lock"
    finally:
        if child is not None:
            stdout, stderr = child.communicate(timeout=10)
            assert child.returncode == 0, f"child failed: {stderr}\n{stdout}"

    assert acquired.read_text(encoding="utf-8") == "True"


def test_dispatch_skips_when_host_cap_lock_cannot_be_acquired(
    kanban_home, all_assignees_spawnable, monkeypatch, caplog,
):
    """A file-lock error must fail closed without claiming or spawning work."""
    spawns: list = []
    with kb.connect() as conn:
        kb.create_task(conn, title="keep-ready", assignee="alice")
        monkeypatch.setattr(
            kb, "_lock_dispatch_host_cap_file",
            lambda handle: (_ for _ in ()).throw(OSError("lock unavailable")),
        )
        result = kb.dispatch_once(
            conn,
            spawn_fn=_fake_spawn_factory(spawns),
            max_in_progress=1,
        )

    assert not spawns
    assert not result.spawned
    assert result.skipped_locked
    assert "could not acquire host-cap lock" in caplog.text


def test_max_spawn_stays_per_board(kanban_home, all_assignees_spawnable):
    """``max_spawn`` keeps its historical per-board semantics."""
    kb.create_board("second")
    with kb.connect(board="second") as conn:
        tid = kb.create_task(conn, title="busy", assignee="alice")
        assert kb.claim_task(conn, tid) is not None

    spawns: list = []
    with kb.connect() as conn:
        kb.create_task(conn, title="a", assignee="alice")
        res = kb.dispatch_once(
            conn, spawn_fn=_fake_spawn_factory(spawns), max_spawn=1,
        )

    # The other board's worker does NOT count against max_spawn.
    assert len(spawns) == 1
    assert len(res.spawned) == 1


# ---------------------------------------------------------------------------
# 3. Review lane cannot be starved by a sustained ready backlog (P2)
# ---------------------------------------------------------------------------


def _park_in_review(conn: sqlite3.Connection, title: str, assignee: str) -> str:
    tid = kb.create_task(conn, title=title, assignee=assignee)
    _set_task_status(conn, tid, "review")
    return tid


def test_review_lane_gets_reserved_slot_under_ready_backlog(
    kanban_home, all_assignees_spawnable, monkeypatch,
):
    import hermes_cli.config as cfgmod
    monkeypatch.setattr(
        cfgmod, "load_config",
        lambda *a, **k: {"kanban": {"review_dispatch": True}},
    )

    spawns: list = []
    with kb.connect() as conn:
        for title in ("ready-1", "ready-2", "ready-3"):
            kb.create_task(conn, title=title, assignee="alice")
        review_id = _park_in_review(conn, "review-me", "reviewer")
        res = kb.dispatch_once(
            conn, spawn_fn=_fake_spawn_factory(spawns), max_in_progress=2,
        )

    spawned_ids = [s[0] for s in res.spawned]
    # Budget 2: one ready + the reserved review slot — never 2×ready.
    assert len(spawned_ids) == 2
    assert review_id in spawned_ids


def test_review_reservation_released_when_no_review_work(
    kanban_home, all_assignees_spawnable, monkeypatch,
):
    import hermes_cli.config as cfgmod
    monkeypatch.setattr(
        cfgmod, "load_config",
        lambda *a, **k: {"kanban": {"review_dispatch": True}},
    )

    spawns: list = []
    with kb.connect() as conn:
        for title in ("ready-1", "ready-2", "ready-3"):
            kb.create_task(conn, title=title, assignee="alice")
        res = kb.dispatch_once(
            conn, spawn_fn=_fake_spawn_factory(spawns), max_in_progress=2,
        )

    # No review work → ready lane keeps the full budget.
    assert len(res.spawned) == 2


def test_nonspawnable_review_does_not_tax_ready_budget(
    kanban_home, monkeypatch,
):
    """Review tasks parked for humans (no real profile) release the slot."""
    import hermes_cli.config as cfgmod
    import hermes_cli.profiles as profmod

    monkeypatch.setattr(
        cfgmod, "load_config",
        lambda *a, **k: {"kanban": {"review_dispatch": True}},
    )
    # Only 'alice' is a real profile; the review assignee is a human lane.
    monkeypatch.setattr(
        profmod, "profile_exists", lambda name: name == "alice"
    )

    spawns: list = []
    with kb.connect() as conn:
        for title in ("ready-1", "ready-2"):
            kb.create_task(conn, title=title, assignee="alice")
        _park_in_review(conn, "human-review", "some-human")
        res = kb.dispatch_once(
            conn, spawn_fn=_fake_spawn_factory(spawns), max_in_progress=2,
        )

    # Human-lane review is not spawnable → no reservation, ready gets both.
    assert len(res.spawned) == 2


def test_review_budget_still_bounded_by_shared_cap(
    kanban_home, all_assignees_spawnable, monkeypatch,
):
    """The reservation caps the ready lane; it grants review no extra slots."""
    import hermes_cli.config as cfgmod
    monkeypatch.setattr(
        cfgmod, "load_config",
        lambda *a, **k: {"kanban": {"review_dispatch": True}},
    )

    spawns: list = []
    with kb.connect() as conn:
        kb.create_task(conn, title="ready-1", assignee="alice")
        for i in range(3):
            _park_in_review(conn, f"review-{i}", "reviewer")
        res = kb.dispatch_once(
            conn, spawn_fn=_fake_spawn_factory(spawns), max_in_progress=2,
        )

    # Budget 2 total across both lanes, reservation notwithstanding.
    assert len(res.spawned) == 2
