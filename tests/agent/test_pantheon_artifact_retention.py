"""Ordinary session maintenance must preserve the logical artifact owner."""
import json
import sqlite3
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_artifacts import ArtifactError
from hermes_state import SessionDB
from agent.message_metadata import REPLY_SOURCE_ROW_ID_KEY
from agent.pantheon_artifacts import commit_response
from tests.agent.test_pantheon_artifacts import runtime, capture, final_messages


def compress(db, parent, child):
    db.end_session(parent, 'compression')
    db.create_session(child, source='pantheon', parent_session_id=parent)


def remove_parent(db, kind):
    if kind=='single': assert db.delete_session('source')
    elif kind=='bulk': assert db.delete_sessions(['source'])==1
    elif kind=='prune': assert db.prune_sessions(older_than_days=None,end_reason='compression')==1
    elif kind=='empty': assert db.delete_empty_sessions()==1
    else: raise AssertionError(kind)


@pytest.mark.parametrize('kind',['single','bulk','prune'])
def test_ready_feed_and_cached_root_survive_ordinary_parent_removal(runtime,kind):
    agent,db,_=runtime
    artifact=capture(runtime);assert commit_response(agent,final_messages())
    descriptor=db.artifact_changes('source','alpha')['changes'][0]['artifact']
    compress(db,'source','tip')
    assert db.artifact_changes('tip','alpha')['conversation_root_id']=='source'
    remove_parent(db,kind)
    assert db.get_session('source') is None
    assert db.get_session('tip')['parent_session_id'] is None
    for alias in ('source','tip'):
        feed=db.artifact_changes(alias,'alpha')
        assert feed['conversation_root_id']=='source'
        assert feed['changes'][0]['artifact']==descriptor
        assert feed['changes'][-1]['type']=='visibility'
        assert feed['changes'][-1]['artifact']['source_visible'] is False
        assert db.artifact_content(alias,artifact)[1].startswith(b'<!doctype')
    assert db._session_turn_lease_key('tip')=='source'
    assert db.artifact_live_session('source')=='tip'
    assert json.loads(db.get_session('tip')['model_config'])['artifact_delivery_version']==1


@pytest.mark.parametrize('kind',['single','bulk','prune'])
def test_accepted_stage_can_commit_in_surviving_compression_tip(runtime,kind):
    agent,db,_=runtime
    artifact=capture(runtime)
    compress(db,'source','tip')
    remove_parent(db,kind)
    agent.session_id='tip';agent._last_flushed_db_idx=0
    assert commit_response(agent,final_messages())
    descriptor=db.artifact_changes('tip','alpha')['changes'][0]['artifact']
    assert descriptor['artifact_id']==artifact
    assert descriptor['conversation_root_id']=='source'
    assert descriptor['source_session_id']=='tip'
    assert db.artifact_content('source',artifact)[1]


def test_empty_parent_cleanup_preserves_tip_publications(runtime):
    agent,db,_=runtime
    compress(db,'source','tip')
    agent.session_id='tip'
    artifact=capture(runtime);assert commit_response(agent,final_messages())
    assert db.delete_empty_sessions()==1
    assert db.get_session('source') is None
    assert db.artifact_changes('tip','alpha')['conversation_root_id']=='source'
    assert db.artifact_content('source',artifact)[1]


@pytest.mark.parametrize('kind',['single','bulk','prune'])
def test_last_segment_removal_erases_registry_and_bytes(runtime,kind):
    agent,db,workspace=runtime
    artifact=capture(runtime);assert commit_response(agent,final_messages())
    assert db.try_acquire_session_turn_lease('source', 'cleanup-holder')
    if kind=='single': assert db.delete_session('source')
    elif kind=='bulk': assert db.delete_sessions(['source'])==1
    else:
        db.end_session('source','done')
        assert db.prune_sessions(older_than_days=None)==1
    assert not db.artifact_directory.joinpath(artifact+'.blob').exists()
    for table in ('pantheon_artifacts','pantheon_artifact_changes','pantheon_artifact_calls','pantheon_artifact_turns'):
        assert db._conn.execute('SELECT COUNT(*) FROM '+table).fetchone()[0]==0
    assert db._conn.execute('SELECT COUNT(*) FROM session_turn_leases').fetchone()[0] == 0
    with pytest.raises(ArtifactError) as gone: db.artifact_changes('source','alpha')
    assert gone.value.status==410
    assert workspace.joinpath('report.html').exists()


def test_empty_cleanup_preserves_staged_resumable_capture(runtime):
    agent,db,_=runtime
    artifact=capture(runtime)
    db.end_session('source','done')
    assert db.count_empty_sessions()==0
    assert db.delete_empty_sessions()==0
    assert db.delete_session_if_empty('source') is False
    assert db.get_session('source') is not None
    assert db.artifact_directory.joinpath(artifact+'.blob').exists()
    assert db.delete_session('source')
    assert not db.artifact_directory.joinpath(artifact+'.blob').exists()


def test_empty_cleanup_preserves_capture_staged_by_removed_ancestor(runtime):
    _, db, _ = runtime
    artifact = capture(runtime)
    compress(db, 'source', 'tip')
    assert db.delete_session('source')
    db.end_session('tip', 'done')
    assert db.count_empty_sessions() == 0
    assert db.delete_empty_sessions() == 0
    assert db.delete_session_if_empty('tip') is False
    db._execute_write(lambda conn: conn.execute(
        "UPDATE sessions SET source='tui',started_at=? WHERE id='tip'", (time.time()-172800,)))
    assert db.prune_empty_ghost_sessions() == 0
    assert db.get_session('tip') is not None
    assert db.artifact_directory.joinpath(artifact+'.blob').exists()


@pytest.mark.parametrize('branch_clone', [False, True])
def test_visibility_requires_surviving_active_clone_in_exact_owner(runtime, branch_clone):
    agent, db, _ = runtime
    artifact = capture(runtime)
    assert commit_response(agent, final_messages())
    source = db.artifact_changes('source', 'alpha')['changes'][0]['artifact']['source_row_id']
    compress(db, 'source', 'tip')
    if branch_clone:
        db.create_session('branch', source='pantheon', parent_session_id='source',
                          model_config={'_branched_from': 'source', 'artifact_delivery_version': 1})
    clone_session = 'branch' if branch_clone else 'tip'
    db.append_message(clone_session, 'assistant', 'Done',
                      display_metadata={REPLY_SOURCE_ROW_ID_KEY: source})
    assert db.delete_session('source')
    changes = db.artifact_changes('tip', 'alpha')['changes']
    assert len(changes) == (2 if branch_clone else 1)
    assert changes[-1]['artifact']['source_visible'] is not branch_clone
    assert db.artifact_content('tip', artifact)[1]
    if branch_clone:
        assert db.artifact_changes('branch', 'alpha')['conversation_root_id'] == 'branch'
        assert db.artifact_changes('branch', 'alpha')['changes'] == []
        with pytest.raises(ArtifactError) as foreign:
            db.artifact_content('branch', artifact)
        assert foreign.value.status == 404


def test_all_segment_bulk_removal_erases_owner_but_preserves_branch_publication(runtime):
    agent, db, _ = runtime
    original = capture(runtime)
    assert commit_response(agent, final_messages())
    compress(db, 'source', 'tip')
    db.create_session('branch', source='pantheon', parent_session_id='source',
                      model_config={'_branched_from': 'source', 'artifact_delivery_version': 1})
    agent.session_id = 'branch'
    agent._current_turn_id = 'branch-turn'
    agent._last_flushed_db_idx = 0
    branch = capture(runtime, call='branch-call')
    assert commit_response(agent, final_messages('Branch answer'))
    assert db.delete_sessions(['source', 'tip']) == 2
    assert not db.artifact_directory.joinpath(original+'.blob').exists()
    assert db.artifact_content('branch', branch)[1]
    assert db.get_session('branch')['parent_session_id'] is None
    assert db.artifact_changes('branch', 'alpha')['conversation_root_id'] == 'branch'
    with pytest.raises(ArtifactError) as gone:
        db.artifact_content('source', original)
    assert gone.value.status == 410


def test_permanent_delete_by_removed_root_alias_erases_survivors_not_branches(runtime):
    agent, db, _ = runtime
    artifact = capture(runtime)
    assert commit_response(agent, final_messages())
    compress(db, 'source', 'tip')
    db.create_session('branch', source='pantheon', parent_session_id='source',
                      model_config={'_branched_from': 'source', 'artifact_delivery_version': 1})
    assert db.delete_session('source')
    result = db.delete_session_permanently('source')
    assert 'tip' in result
    assert db.get_session('tip') is None
    assert db.get_session('branch') is not None
    assert not db.artifact_directory.joinpath(artifact+'.blob').exists()
    assert db._conn.execute('SELECT COUNT(*) FROM pantheon_artifact_turns').fetchone()[0] == 0
    owner = db._conn.execute("SELECT erased,cursor_key FROM pantheon_artifact_owners WHERE root='source'").fetchone()
    assert tuple(owner) == (1, '')
    with pytest.raises(ArtifactError) as gone:
        db.artifact_changes('source', 'alpha')
    assert gone.value.status == 410


def test_atomic_compression_publication_registers_owner_and_capability(runtime):
    agent, db, _ = runtime
    artifact = capture(runtime)
    assert commit_response(agent, final_messages())
    messages = db.get_messages_as_conversation('source', include_row_ids=True)
    db.publish_compression_child(parent_session_id='source', child_session_id='tip',
                                 source='pantheon', messages=messages,
                                 require_compression_lease=False)
    assert json.loads(db.get_session('tip')['model_config'])['artifact_delivery_version'] == 1
    assert db.artifact_conversation_root('tip') == 'source'
    assert db.delete_session('source')
    assert db.artifact_content('tip', artifact)[1]
    assert db.artifact_changes('tip', 'alpha')['changes'][-1]['artifact']['source_visible']


def test_existing_version28_owner_is_backfilled_before_parent_detachment(runtime):
    agent, db, _ = runtime
    artifact = capture(runtime)
    assert commit_response(agent, final_messages())
    compress(db, 'source', 'tip')
    # Existing version-28 registry tables have no segment aliases yet.
    db._execute_write(lambda conn: conn.execute('DELETE FROM pantheon_artifact_segments'))
    assert db.delete_session('source')
    with SessionDB(db.db_path) as reopened:
        assert reopened.artifact_changes('tip', 'alpha')['conversation_root_id'] == 'source'
        assert reopened.artifact_content('source', artifact)[1]


def test_empty_ghost_parent_prune_keeps_child_owner(runtime):
    agent, db, _ = runtime
    compress(db, 'source', 'tip')
    agent.session_id = 'tip'
    artifact = capture(runtime)
    assert commit_response(agent, final_messages())
    db._execute_write(lambda conn: conn.execute(
        "UPDATE sessions SET source='tui',started_at=? WHERE id='source'", (time.time()-172800,)))
    assert db.prune_empty_ghost_sessions() == 1
    assert db.artifact_content('source', artifact)[1]
    assert db.artifact_conversation_root('tip') == 'source'


def test_empty_last_segment_cleanup_erases_owner_fence(runtime):
    _, db, _ = runtime
    assert db.delete_session_if_empty('source')
    owner = db._conn.execute("SELECT erased,cursor_key FROM pantheon_artifact_owners WHERE root='source'").fetchone()
    assert tuple(owner) == (1, '')
    assert db.artifact_live_session('source') is None


def test_deletion_hook_rolls_back_registry_visibility_and_bytes_with_session(runtime):
    agent, db, _ = runtime
    artifact = capture(runtime)
    assert commit_response(agent, final_messages())
    compress(db, 'source', 'tip')
    db._execute_write(lambda conn: conn.execute(
        "CREATE TRIGGER fail_delete BEFORE DELETE ON sessions BEGIN SELECT RAISE(ABORT,'fixture failure'); END"))
    with pytest.raises(sqlite3.IntegrityError, match='fixture failure'):
        db.delete_session('source')
    assert db.get_session('source') is not None
    changes = db.artifact_changes('tip', 'alpha')['changes']
    assert len(changes) == 1
    assert changes[0]['artifact']['source_visible']
    assert db.artifact_content('source', artifact)[1]


def test_failed_ordinary_delete_unlink_retries_even_after_session_absent(runtime, monkeypatch):
    agent, db, _ = runtime
    artifact = capture(runtime)
    assert commit_response(agent, final_messages())
    blob = db.artifact_directory / (artifact+'.blob')
    original = Path.unlink
    def fail_blob(path, *args, **kwargs):
        if path == blob:
            raise PermissionError('fixture unlink failure')
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'unlink', fail_blob)
    with pytest.raises(PermissionError, match='fixture unlink failure'):
        db.delete_session('source')
    assert db.get_session('source') is None
    assert blob.exists()
    assert db._conn.execute('SELECT COUNT(*) FROM pantheon_artifact_cleanup').fetchone()[0] == 1
    monkeypatch.setattr(Path, 'unlink', original)
    assert db.delete_session('source') is False
    assert not blob.exists()
    assert db._conn.execute('SELECT COUNT(*) FROM pantheon_artifact_cleanup').fetchone()[0] == 0


def test_actual_authenticated_rest_cached_root_reads_and_delete_retry(runtime, monkeypatch):
    from starlette.testclient import TestClient
    from hermes_cli import web_server as web
    agent, db, _ = runtime
    artifact = capture(runtime)
    assert commit_response(agent, final_messages())
    messages = db.get_messages_as_conversation('source', include_row_ids=True)
    db.publish_compression_child(parent_session_id='source', child_session_id='tip',
                                 source='pantheon', messages=messages,
                                 require_compression_lease=False)
    monkeypatch.setattr(web, '_open_session_db_for_profile',
                        lambda *args, **kwargs: SessionDB(db.db_path))
    monkeypatch.setattr(web, '_cron_default_profile', lambda: 'alpha')
    client = TestClient(web.app)
    client.headers[web._SESSION_HEADER_NAME] = web._SESSION_TOKEN
    removed = client.delete('/api/sessions/source')
    assert removed.status_code == 200, removed.text
    # A retained exact root alias is never an abbreviation of a new row.
    db.create_session('source-unrelated', source='pantheon')
    feed = client.get('/api/sessions/source/artifacts')
    assert feed.status_code == 200, feed.text
    assert feed.json()['conversation_root_id'] == 'source'
    content = client.get(f'/api/sessions/source/artifacts/{artifact}/content')
    assert content.status_code == 200, content.text
    latest = client.get('/api/sessions/source/latest-descendant')
    assert latest.status_code == 200, latest.text
    assert latest.json()['session_id'] == 'tip'
    detail = client.get('/api/sessions/source')
    assert detail.status_code == 200, detail.text
    assert detail.json()['id'] == 'tip'
    history = client.get('/api/sessions/source/messages')
    assert history.status_code == 200, history.text
    assert any(row['content'] == 'Done' for row in history.json()['messages'])
    retried = client.delete('/api/sessions/source')
    assert retried.status_code == 200, retried.text
    assert retried.json()['already_absent']
    assert db.get_session('tip') is not None
    assert db.get_session('source-unrelated') is not None
    rejected = TestClient(web.app).get(f'/api/sessions/source/artifacts/{artifact}/content')
    assert rejected.status_code == 401
    client.close()


def test_actual_cold_gateway_resume_resolves_removed_root_and_original_capability(runtime, monkeypatch):
    import tui_gateway.server as gateway
    from tests.tui_gateway.test_profiles_list_canonical_session import _resume
    _, db, _ = runtime
    compress(db, 'source', 'tip')
    assert db.delete_session('source')
    monkeypatch.setattr(gateway, '_get_db', lambda: db)
    result = _resume({'session_id': 'source', 'artifact_delivery_version': 1}, monkeypatch)
    assert 'error' not in result, result
    assert result['result']['resumed'] == 'tip'
    assert result['result']['artifact_delivery_version'] == 1
    assert result['result']['conversation_root_id'] == 'source'
