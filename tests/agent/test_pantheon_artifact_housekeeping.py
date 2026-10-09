"""Shared RPC backends recover each existing local profile independently."""
import copy
import sqlite3
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import hermes_artifacts as artifacts
from hermes_state import SessionDB
from agent.pantheon_artifacts import commit_response
from tests.agent.test_pantheon_artifacts import runtime, capture, final_messages


def profile_capture(runtime, home, name):
    original, _, workspace = runtime
    db = SessionDB(home/'profiles'/name/'state.db')
    db.create_session('source', source='pantheon', model_config={'artifact_delivery_version': 1})
    agent = copy.copy(original)
    agent._session_db = db
    agent._pantheon_profile = name
    agent._pantheon_profile_home = str(db.db_path.parent)
    agent._last_flushed_db_idx = 0
    artifact = capture((agent, db, workspace))
    return agent, db, artifact


def test_shared_backend_housekeeping_collects_dormant_profile_and_preserves_ready_and_live(runtime, monkeypatch, tmp_path):
    home = tmp_path/'catalog'
    monkeypatch.setenv('HERMES_HOME', str(home))
    agent, dormant, abandoned = profile_capture(runtime, home, 'dormant')
    dormant.artifact_finish_call('source', 'turn-a', 'call-a', False)
    dormant._execute_write(lambda conn: conn.execute(
        'UPDATE pantheon_artifacts SET abandoned_at=?', (time.time()-90000,)))
    agent, ready, ready_id = profile_capture(runtime, home, 'ready')
    assert commit_response(agent, final_messages())
    _, live, live_id = profile_capture(runtime, home, 'live')
    paths = {str(db.db_path) for db in (dormant, ready, live)}
    for db in (dormant, ready, live):
        db.close()
    results = artifacts.recover_profile_artifact_captures()
    assert set(results) == paths
    assert results[str(dormant.db_path)] == 1
    assert not dormant.artifact_directory.joinpath(abandoned+'.blob').exists()
    assert ready.artifact_directory.joinpath(ready_id+'.blob').exists()
    assert live.artifact_directory.joinpath(live_id+'.blob').exists()
    with SessionDB(live.db_path) as reopened:
        assert reopened._conn.execute('SELECT state FROM pantheon_artifacts').fetchone()[0] == 'staged'
    assert not home.joinpath('state.db').exists()  # no default bootstrap


def test_profile_failure_isolated_and_missing_deleted_or_symlinked_catalog_entries_skipped(runtime, monkeypatch, tmp_path, caplog):
    from hermes_constants import mark_named_profile_deleted
    home = tmp_path/'catalog'
    monkeypatch.setenv('HERMES_HOME', str(home))
    profiles = {}
    for name in ('broken', 'healthy', 'deleted', 'artifact-link'):
        _, db, artifact = profile_capture(runtime, home, name)
        profiles[name] = (db, artifact)
        db.close()
    mark_named_profile_deleted(home/'profiles'/'deleted')
    (home/'profiles'/'missing').mkdir()
    (home/'profiles'/'directory-link').symlink_to(home/'profiles'/'healthy', target_is_directory=True)
    (home/'profiles'/'db-link').mkdir()
    (home/'profiles'/'db-link'/'state.db').symlink_to(profiles['healthy'][0].db_path)
    linked = home/'profiles'/'artifact-link'/'pantheon-artifacts'
    linked.rename(linked.with_name('saved-artifacts'))
    linked.symlink_to(linked.with_name('saved-artifacts'), target_is_directory=True)
    original = SessionDB.recover_artifact_captures
    def fail_one(db, *args, **kwargs):
        if db.db_path == profiles['broken'][0].db_path:
            raise PermissionError('do not log /raw/private/fixture/path')
        return original(db, *args, **kwargs)
    monkeypatch.setattr(SessionDB, 'recover_artifact_captures', fail_one)
    results = artifacts.recover_profile_artifact_captures()
    assert set(results) == {str(profiles['healthy'][0].db_path)}
    assert 'PermissionError' in caplog.text
    assert '/raw/private/fixture/path' not in caplog.text
    assert not home.joinpath('profiles', 'missing', 'state.db').exists()


def test_supplied_launch_handle_is_not_closed_or_reopened(runtime, monkeypatch, tmp_path):
    _, db, _ = runtime
    monkeypatch.setenv('HERMES_HOME', str(tmp_path/'empty-catalog'))
    assert str(db.db_path) in artifacts.recover_profile_artifact_captures(db)
    assert db.get_session('source') is not None


def test_existing_only_database_open_refuses_missing_without_mkdir(tmp_path):
    path = tmp_path/'missing-profile'/'state.db'
    with pytest.raises(FileNotFoundError):
        SessionDB(path, create_if_missing=False)
    assert not path.parent.exists()


def test_existing_only_mode_rw_refuses_removed_database_at_connect(tmp_path, monkeypatch):
    import hermes_state
    path = tmp_path/'profile'/'state.db'
    SessionDB(path).close()
    original = hermes_state._connect_tracked_db
    def remove_before_connect(database, **kwargs):
        assert kwargs.get('uri') is True
        assert str(database).endswith('?mode=rw')
        path.unlink()
        return original(database, **kwargs)
    monkeypatch.setattr(hermes_state, '_connect_tracked_db', remove_before_connect)
    with pytest.raises(sqlite3.OperationalError):
        SessionDB(path, create_if_missing=False)
    assert not path.exists()


def test_housekeeping_loop_runs_at_startup_and_hourly_without_rpc(monkeypatch):
    calls = []
    monkeypatch.setattr(artifacts, 'recover_profile_artifact_captures', lambda: calls.append('recover'))
    class Stop:
        count = 0
        def is_set(self):
            return self.count == 2
        def wait(self, seconds):
            assert seconds == 3600
            self.count += 1
    artifacts._profile_artifact_housekeeping_loop(Stop())
    assert calls == ['recover', 'recover']


def test_housekeeping_starts_once_and_is_independent_of_heartbeat_opt_out(monkeypatch):
    from tui_gateway import server
    calls = []
    class Thread:
        def __init__(self, **kwargs):
            assert kwargs['daemon']
            self.target = kwargs['target']
        def start(self):
            calls.append(self.target)
        def is_alive(self):
            return True
    monkeypatch.setattr(artifacts, '_housekeeping_thread', None)
    monkeypatch.setattr(artifacts.threading, 'Thread', Thread)
    monkeypatch.setattr(server, '_heartbeat_refresher_started', False)
    monkeypatch.setattr(server, '_HEARTBEAT_REFRESH_S', 0)
    monkeypatch.setattr(server, '_refresh_backend_heartbeat', lambda: None)
    server._start_backend_heartbeat_refresher()
    assert calls == [artifacts._profile_artifact_housekeeping_loop]
    assert artifacts.start_profile_artifact_housekeeping() is False


@pytest.mark.parametrize('failure', ['deleted', 'database'])
def test_stop_propagates_every_runtime_fence_when_artifact_cancellation_fails(runtime, monkeypatch, failure):
    import run_agent
    agent, db, _ = runtime
    capture(runtime)
    agent.api_mode = 'codex_app_server'
    propagated = []
    agent._codex_session = SimpleNamespace(request_interrupt=lambda: propagated.append('codex'))
    agent._active_request_abort = lambda reason: propagated.append('socket')
    agent._execution_thread_id = 11
    agent._tool_worker_threads = {12}
    agent._tool_worker_threads_lock = threading.Lock()
    agent._active_children_lock = threading.Lock()
    agent._active_children = [SimpleNamespace(hard_interrupt=lambda *args, **kwargs: propagated.append('child'))]
    agent._hard_interrupt_requested = threading.Event()
    agent.quiet_mode = True
    monkeypatch.setattr(run_agent, '_set_interrupt', lambda value, tid, **kwargs: propagated.append(tid))
    if failure == 'deleted':
        db.delete_session('source')
    else:
        def fail_persistence(*args, **kwargs):
            assert agent._interrupt_requested
            assert agent._hard_interrupt_requested.is_set()
            assert set(propagated) == {'codex', 'socket', 11, 12, 'child'}
            raise OSError('fixture database unavailable')
        monkeypatch.setattr(db, 'artifact_cancel_turn', fail_persistence)
    assert agent.interrupt(hard_cancel=True)
    assert agent._interrupt_requested
    assert agent._hard_interrupt_requested.is_set()
    assert set(propagated) == {'codex', 'socket', 11, 12, 'child'}
    if failure == 'database':
        assert commit_response(agent, final_messages()) is False
        # Clearing an ordinary runtime interrupt cannot revive the same turn.
        agent._interrupt_requested = False
        assert commit_response(agent, final_messages()) is False
        assert db.artifact_changes('source', 'alpha')['changes'] == []
