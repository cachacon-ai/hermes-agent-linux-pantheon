"""Publication invariants exercise real filesystem + SessionDB transactions."""
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from hermes_artifacts import ArtifactError, MAX_ARTIFACT_BYTES
from hermes_state import SessionDB
from run_agent import AIAgent
from agent.pantheon_artifacts import (safe_source_bytes, checked_mime, dispatch,
    accept_result, commit_response, cancel_turn, publish)


@pytest.fixture
def runtime(tmp_path):
    workspace=tmp_path/'workspace';workspace.mkdir()
    db=SessionDB(tmp_path/'profile'/'state.db')
    db.create_session('source',source='pantheon',model_config={'artifact_delivery_version':1})
    agent=AIAgent.__new__(AIAgent)
    agent._session_db=db;agent._session_db_created=True
    agent._last_flushed_db_idx=0;agent.session_id='source'
    agent._current_turn_id='turn-a';agent._current_task_id='task-a'
    agent._pantheon_artifact_delivery_version=1;agent._pantheon_profile='alpha'
    agent._pantheon_profile_home=str(tmp_path/'profile')
    agent._pantheon_artifact_roots=(str(workspace),)
    agent._interrupt_requested=False
    agent._active_session_turn_lease_holder=None
    yield agent,db,workspace
    db.close()


def capture(runtime,call='call-a',data=b'<!doctype html><html><button>ok</button></html>'):
    agent,db,workspace=runtime
    source=workspace/'report.html';source.write_bytes(data)
    result=dispatch(agent,call,'task-a',{'path':str(source)},native_environment=SimpleNamespace(cwd=str(workspace)))
    assert json.loads(result).get('status')=='captured',result
    accept_result(agent,call,result)
    return json.loads(result)['artifact_id']


def final_messages(content='Done'):
    return [{'role':'user','content':'generate','timestamp':1780000000.0}, {'role':'assistant','content':content,'timestamp':1780000001.0,'display_metadata':{'reactions':[{'emoji':'ok'}]}}]


def test_capture_not_visible_until_exact_response_commit_and_durable_reopen(runtime):
    agent,db,workspace=runtime
    artifact=capture(runtime)
    assert db.artifact_changes('source','alpha')['changes']==[]
    with pytest.raises(ArtifactError,match='not committed'):
        db.artifact_content('source',artifact)
    messages=final_messages();assert commit_response(agent,messages) is True
    rows=db.get_messages_as_conversation('source',include_row_ids=True)
    descriptor=db.artifact_changes('source','alpha')['changes'][0]['artifact']
    assert descriptor['source_row_id']==rows[-1]['_row_id']
    assert rows[-1]['display_metadata']['reactions']==[{'emoji':'ok'}]
    assert rows[-1]['display_metadata']['pantheon_artifacts']==[descriptor]
    (workspace/'report.html').write_text('overwritten')
    assert db.artifact_content('source',artifact)[1].startswith(b'<!doctype')
    with SessionDB(db.db_path) as reopened:
        assert reopened.artifact_changes('source','alpha')['changes'][0]['artifact']==descriptor


def test_early_flushed_final_and_zero_fresh_batch_binds_atomically_once(runtime):
    agent,db,_=runtime;artifact=capture(runtime)
    messages=final_messages();agent._flush_messages_to_session_db(messages)
    source_id=messages[-1]['_row_id']
    assert commit_response(agent,messages)
    assert commit_response(agent,messages)
    assert len(db.artifact_changes('source','alpha')['changes'])==1
    assert db.artifact_changes('source','alpha')['changes'][0]['artifact']['source_row_id']==source_id


def test_fileonly_uses_display_sidecar_never_provider_content(runtime):
    agent,db,_=runtime;capture(runtime)
    messages=final_messages('');assert commit_response(agent,messages)
    assert messages[-1]['content']==''
    assert messages[-1]['display_metadata']['pantheon_reply_preview']=='Generated report.html.'
    from tui_gateway.server import _history_to_messages, _persisted_reply_identity
    projected=_history_to_messages(messages)
    assert projected[-1]['text']=='Generated report.html.'
    assert _persisted_reply_identity({'messages':messages,'reply_row_id':messages[-1]['_row_id']})['reply_text']=='Generated report.html.'


def test_failure_rollback_does_not_stamp_or_ready(runtime,monkeypatch):
    agent,db,_=runtime;capture(runtime)
    original=db._artifact_commit_on_conn
    def fail(conn,sid,fin):
        original(conn,sid,fin)
        raise OSError('disk full')
    monkeypatch.setattr(db,'_artifact_commit_on_conn',fail)
    messages=final_messages()
    assert commit_response(agent,messages) is False
    assert not messages[-1].get('_db_persisted')
    assert db.get_messages('source')==[]
    assert db.artifact_changes('source','alpha')['changes']==[]
    monkeypatch.setattr(db,'_artifact_commit_on_conn',original)
    assert commit_response(agent,messages)
    assert len(db.artifact_changes('source','alpha')['changes'])==1


def test_rollback_cannot_leak_canonical_identity_into_retry_after_other_writer(runtime, monkeypatch):
    agent, db, _ = runtime
    capture(runtime)
    original = db._artifact_commit_on_conn
    def fail_after_binding(conn, sid, finalization):
        original(conn, sid, finalization)
        raise OSError('fixture rollback after artifact binding')
    monkeypatch.setattr(db, '_artifact_commit_on_conn', fail_after_binding)
    messages = final_messages()
    assert commit_response(agent, messages) is False
    assert messages[-1]['display_metadata'] == {'reactions': [{'emoji': 'ok'}]}
    assert '_row_id' not in messages[-1]
    # Another writer consumes the ids from the rolled-back transaction.
    db.create_session('unrelated', source='cli')
    unrelated = db.append_message('unrelated', 'assistant', 'Other response')
    db.append_message('unrelated', 'user', 'Other question')
    monkeypatch.setattr(db, '_artifact_commit_on_conn', original)
    assert commit_response(agent, messages)
    descriptor = db.artifact_changes('source', 'alpha')['changes'][0]['artifact']
    assert descriptor['source_row_id'] == messages[-1]['_row_id']
    assert descriptor['source_row_id'] > unrelated
    assert messages[-1]['display_metadata']['reactions'] == [{'emoji': 'ok'}]


def test_cancel_and_executor_revocation_prevent_late_capture(runtime):
    agent,db,workspace=runtime
    source=workspace/'x.txt';source.write_text('ok')
    accept_result(agent,'late','',abandoned=True)
    result=dispatch(agent,'late','task-a',{'path':str(source)},native_environment=SimpleNamespace(cwd=str(workspace)))
    assert json.loads(result)['error']=='invocation_revoked'
    artifact=capture(runtime)
    cancel_turn(agent)
    assert commit_response(agent,final_messages()) is False
    assert db.artifact_changes('source','alpha')['changes']==[]
    assert db.cleanup_abandoned_artifacts(now=10**11)==1
    assert list(db.artifact_directory.glob('*.blob'))==[]


def test_tool_json_prose_and_wrong_context_cannot_publish(runtime):
    agent,db,workspace=runtime
    source=workspace/'x.txt';source.write_text('ok')
    assert json.loads(publish({'path':str(source)}))['error']=='missing_context'
    agent._pantheon_artifact_delivery_version=0
    assert json.loads(dispatch(agent,'call','task',{'path':str(source)}))['error']=='capability_unavailable'
    assert db.artifact_changes('source','alpha')['changes']==[]


def test_unsupported_environment_never_falls_back_to_host(runtime,monkeypatch):
    agent,db,workspace=runtime
    source=workspace/'x.txt';source.write_text('host secret')
    monkeypatch.setattr('tools.terminal_tool.get_active_env',lambda task:SimpleNamespace(cwd=str(workspace)))
    assert json.loads(dispatch(agent,'call','task',{'path':str(source)}))['error']=='unsupported_source'
    assert not db.artifact_directory.exists()


@pytest.mark.parametrize('case',['leaf_symlink','parent_symlink','fifo','directory','outside','protected','oversize'])
def test_posix_reader_rejects_unsafe_sources(tmp_path,case):
    root=tmp_path/'root';root.mkdir();path=root/'file.txt';path.write_text('ok')
    if case=='leaf_symlink': path.unlink();path.symlink_to('/etc/hosts')
    if case=='parent_symlink':
        (root/'link').symlink_to(tmp_path,target_is_directory=True);path=root/'link'/'outside.txt'
    if case=='fifo': path.unlink();os.mkfifo(path)
    if case=='directory': path=root
    if case=='outside': path=tmp_path/'outside.txt';path.write_text('secret')
    if case=='protected': path=root/'.env';path.write_text('secret')
    if case=='oversize':
        with path.open('wb') as f:f.truncate(MAX_ARTIFACT_BYTES+1)
    with pytest.raises(ArtifactError):safe_source_bytes(str(path),(str(root),))


def test_reader_rejects_during_read_mutation(tmp_path,monkeypatch):
    root=tmp_path/'root';root.mkdir();path=root/'x.txt';path.write_bytes(b'a'*100)
    original=os.read
    def mutate(fd,count):
        data=original(fd,count)
        path.write_bytes(b'b'*100)
        return data
    monkeypatch.setattr(os,'read',mutate)
    with pytest.raises(ArtifactError,match='changed'):safe_source_bytes(str(path),(str(root),))


@pytest.mark.parametrize('name,data,mime',[
 ('report.html',b'<!doctype html><html></html>','text/html'),
 ('x.png',b'\x89PNG\r\n\x1a\n','image/png'),('x.pdf',b'%PDF-1.4','application/pdf'),
 ('x.txt',b'hello','text/plain')])
def test_content_type_policy(name,data,mime):assert checked_mime(name,data)==mime


@pytest.mark.parametrize('name,data',[('x.png',b'<html>'),('x.svg',b'<svg/>'),('x.exe',b'MZ'),('x.zip',b'PK\x03\x04'),('x.html',b'\xff')])
def test_content_type_mismatch_and_active_formats(name,data):
    with pytest.raises(ArtifactError):checked_mime(name,data)


def test_pinned_changefeed_and_owner_scoped_cursor(runtime):
    agent,db,_=runtime
    for i in range(105): capture(runtime,call='call-'+str(i))
    assert commit_response(agent,final_messages())
    page=db.artifact_changes('source','alpha');assert len(page['changes'])==100
    assert page['next_cursor']
    second=db.artifact_changes('source','alpha',cursor=page['next_cursor'])
    assert len(second['changes'])==5 and second['through_revision']==page['through_revision']
    with pytest.raises(ArtifactError):db.artifact_changes('source','other',cursor=page['next_cursor'])
    db.create_session('branch',source='pantheon',parent_session_id='source',model_config={'_branched_from':'source'})
    assert db.artifact_changes('branch','alpha')['changes']==[]
    with pytest.raises(ArtifactError):db.artifact_changes('branch','alpha',cursor=page['next_cursor'])
    with pytest.raises(ArtifactError) as foreign:db.artifact_content('branch',page['changes'][0]['artifact']['artifact_id'])
    assert foreign.value.status==404


def test_rewind_hides_source_and_file_deletion_tombstone_is_durable(runtime):
    agent,db,workspace=runtime;artifact=capture(runtime)
    messages=final_messages();commit_response(agent,messages)
    db.rewind_to_message('source',messages[0]['_row_id'])
    changes=db.artifact_changes('source','alpha')['changes']
    assert [c['type'] for c in changes]==['ready','visibility']
    assert changes[-1]['artifact']['source_visible'] is False
    assert db.artifact_content('source',artifact)[1]
    db.artifact_delete('source',artifact);db.artifact_delete('source',artifact)
    assert db.artifact_changes('source','alpha')['changes'][-1]['artifact']=={'artifact_id':artifact,'profile':'alpha','conversation_root_id':'source'}
    with pytest.raises(ArtifactError) as missing:db.artifact_content('source',artifact)
    assert missing.value.status==410
    assert (workspace/'report.html').exists()


def test_permanent_erasure_purges_feed_payload_and_invalidates_cursor(runtime):
    agent,db,workspace=runtime;artifact=capture(runtime);commit_response(agent,final_messages())
    db.delete_session_permanently('source')
    for table in ('pantheon_artifacts','pantheon_artifact_changes','pantheon_artifact_calls','pantheon_artifact_turns'):
        assert db._conn.execute('SELECT COUNT(*) FROM '+table).fetchone()[0]==0
    assert list(db.artifact_directory.glob('*.blob'))==[]
    assert (workspace/'report.html').exists()
    with pytest.raises(ArtifactError) as erased:db.artifact_changes('source','alpha')
    assert erased.value.status==410


def test_default_session_turn_output_is_usable_and_separate_each_turn(runtime):
    from agent.pantheon_artifacts import prepare_turn_output
    agent,db,workspace=runtime
    agent._pantheon_artifact_roots=()
    note=prepare_turn_output(agent)
    root=Path(agent._pantheon_turn_output_root)
    assert str(root) in note
    source=root/'default.txt';source.write_text('deliverable')
    result=dispatch(agent,'default-call','task-a',{'path':str(source)},native_environment=SimpleNamespace(cwd=str(workspace)))
    assert json.loads(result)['status']=='captured'
    assert json.loads(result)['delivered'] is False
    accept_result(agent,'default-call',result)
    assert commit_response(agent,final_messages())
    agent._current_turn_id='turn-b'
    prepare_turn_output(agent)
    assert agent._pantheon_turn_output_root!=str(root)
    result=dispatch(agent,'new-call','task-a',{'path':str(source)},native_environment=SimpleNamespace(cwd=str(workspace)))
    assert json.loads(result)['error']=='source_outside_root'


def test_snapshot_cleanup_retries_after_permanent_delete_unlink_failure(runtime,monkeypatch):
    agent,db,workspace=runtime;artifact=capture(runtime);commit_response(agent,final_messages())
    real_unlink=Path.unlink
    failed=[False]
    def unlink(path,*args,**kwargs):
        if path.suffix=='.blob' and not failed[0]:
            failed[0]=True
            raise OSError('busy filesystem')
        return real_unlink(path,*args,**kwargs)
    monkeypatch.setattr(Path,'unlink',unlink)
    with pytest.raises(OSError):db.delete_session_permanently('source')
    assert db._conn.execute('SELECT COUNT(*) FROM pantheon_artifact_cleanup').fetchone()[0]==1
    assert db.artifact_directory.joinpath(artifact+'.blob').exists()
    db.delete_session_permanently('source')
    assert db._conn.execute('SELECT COUNT(*) FROM pantheon_artifact_cleanup').fetchone()[0]==0
    assert not db.artifact_directory.joinpath(artifact+'.blob').exists()


def test_retry_same_invocation_returns_original_snapshot_after_source_overwrite(runtime):
    agent,db,workspace=runtime
    artifact=capture(runtime)
    (workspace/'report.html').write_text('not html anymore')
    result=dispatch(agent,'call-a','task-a',{'path':str(workspace/'report.html')},native_environment=SimpleNamespace(cwd=str(workspace)))
    assert json.loads(result)['artifact_id']==artifact
    assert commit_response(agent,final_messages())
    assert len(db.artifact_changes('source','alpha')['changes'])==1
    assert db.artifact_content('source',artifact)[1].startswith(b'<!doctype')


def test_frozen_invocation_cannot_move_into_next_turn(runtime,monkeypatch):
    from agent.pantheon_artifacts import freeze_authority
    from tools.environments.local import LocalEnvironment
    agent,db,workspace=runtime
    environment=LocalEnvironment.__new__(LocalEnvironment);environment.cwd=str(workspace)
    monkeypatch.setattr('tools.terminal_tool.get_active_env',lambda task:environment)
    source=workspace/'late.txt';source.write_text('frozen')
    frozen=freeze_authority(agent,'task-a')
    accept_result(agent,'late','',abandoned=True,authority=frozen)
    agent._current_turn_id='turn-b';agent._pantheon_profile='beta'
    result=dispatch(agent,'late','task-a',{'path':str(source)},authority=frozen)
    assert json.loads(result)['error']=='invocation_revoked'
    assert db._conn.execute('SELECT COUNT(*) FROM pantheon_artifacts').fetchone()[0]==0
    assert db._conn.execute("SELECT COUNT(*) FROM pantheon_artifact_turns WHERE turn_id='turn-b'").fetchone()[0]==0


def test_compression_clone_preserves_registry_descriptor_identity(runtime):
    from agent.context_compressor import stamp_db_persisted_markers
    agent,db,_=runtime;capture(runtime)
    messages=final_messages();assert commit_response(agent,messages)
    before=db.artifact_changes('source','alpha')['changes'][0]['artifact']
    copies=[dict(m) for m in messages]
    db.archive_and_compact('source',copies,tail_count=len(copies))
    stamp_db_persisted_markers(copies)
    assert copies[-1]['_row_id'] != messages[-1]['_row_id']
    assert commit_response(agent,copies)
    assert agent._pantheon_committed_artifacts==[before]
    assert len(db.artifact_changes('source','alpha')['changes'])==1


def test_rejected_capture_is_collected_after_successful_text_commit(runtime):
    agent,db,_=runtime
    artifact=capture(runtime)
    db.artifact_finish_call('source','turn-a','call-a',False)
    assert commit_response(agent,final_messages())
    assert db.artifact_changes('source','alpha')['changes']==[]
    row=db._conn.execute('SELECT abandoned_at FROM pantheon_artifacts WHERE artifact_id=?',(artifact,)).fetchone()
    assert row[0] is not None
    assert db.cleanup_abandoned_artifacts(now=row[0]+23*3600)==0
    assert db.artifact_directory.joinpath(artifact+'.blob').exists()
    assert db.cleanup_abandoned_artifacts(now=row[0]+25*3600)==1
    assert not db.artifact_directory.joinpath(artifact+'.blob').exists()


def test_recovery_preserves_expired_live_holder_but_fences_superseded_turn(runtime):
    import time
    agent,db,_=runtime
    holder='pid='+str(os.getpid())+':live'
    assert db.try_acquire_session_turn_lease('source',holder)
    agent._active_session_turn_lease_holder=holder
    artifact=capture(runtime)
    db._execute_write(lambda conn:conn.execute('UPDATE session_turn_leases SET expires_at=0'))
    assert db.recover_artifact_captures(now=time.time()+10)==0
    assert db._conn.execute('SELECT state FROM pantheon_artifact_turns').fetchone()[0]=='active'
    assert db.artifact_directory.joinpath(artifact+'.blob').exists()
    db._execute_write(lambda conn:conn.execute("UPDATE session_turn_leases SET holder='pid=999999999:new'"))
    now=time.time()+20
    assert db.recover_artifact_captures(now=now)==0
    assert db._conn.execute('SELECT state FROM pantheon_artifact_turns').fetchone()[0]=='abandoned'
    assert db.recover_artifact_captures(now=now+25*3600)==1
    assert not db.artifact_directory.joinpath(artifact+'.blob').exists()


def test_orphan_recovery_observes_capture_fence_and_24h_grace(runtime):
    import time
    from hermes_artifacts import capture_lock
    agent,db,_=runtime
    directory=db.artifact_directory
    with capture_lock(directory):
        orphan=directory/'00000000-0000-0000-0000-000000000000.blob'
        orphan.write_bytes(b'orphan')
        now=time.time()
        assert db.recover_artifact_captures(now=now)==0
        assert db._conn.execute('SELECT COUNT(*) FROM pantheon_artifact_orphans').fetchone()[0]==0
    assert db.recover_artifact_captures(now=now+1)==0
    assert db.recover_artifact_captures(now=now+23*3600)==0
    assert orphan.exists()
    assert db.recover_artifact_captures(now=now+25*3600)==1
    assert not orphan.exists()


@pytest.mark.parametrize('kind,code',[('remote','unsupported_source'),('native','unsupported_protocol'),('invalid','invalid_configuration'),('local',None)])
def test_capability_reports_actionable_unavailability(runtime,monkeypatch,kind,code):
    from agent.pantheon_artifacts import capability_details
    agent,db,_=runtime
    home=Path(agent._pantheon_profile_home)
    home.mkdir(exist_ok=True)
    config={'remote':'terminal:\n  backend: docker\n','native':'model:\n  openai_runtime: codex_app_server\n','invalid':'[bad','local':'terminal:\n  backend: local\n'}[kind]
    (home/'config.yaml').write_text(config)
    monkeypatch.delenv('TERMINAL_ENV',raising=False)
    monkeypatch.setattr('agent.transports.codex_app_server_session.supports_pantheon_dynamic_tools',lambda *_:False)
    result=capability_details(home)
    assert result['artifact_delivery_version']==(1 if code is None else 0)
    if code: assert result['artifact_delivery_unavailable_code']==code


def test_capability_projection_and_history_keep_provider_text_separate(runtime,monkeypatch):
    from contextlib import nullcontext
    from tui_gateway import server
    monkeypatch.setattr(server,'_session_db',lambda session:nullcontext(None))
    record={'artifact_delivery_version':0,'artifact_profile':'alpha','session_key':'source',
        'artifact_delivery_unavailable_code':'unsupported_protocol','artifact_delivery_unavailable_reason':'Adapter update required.'}
    projected=server._artifact_capability_payload(record)
    assert projected['artifact_delivery_unavailable_code']=='unsupported_protocol'
    assert projected['profile']=='alpha'
    agent,db,_=runtime;capture(runtime)
    messages=final_messages('');assert commit_response(agent,messages)
    reply=server._history_to_messages(messages)[-1]
    assert reply['artifact_only'] is True and reply['provider_text']==''
    assert reply['presentation_text']=='Generated report.html.'


@pytest.mark.parametrize('extension,primary,content_type',[
    ('.docx','word/document.xml','application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml'),
    ('.xlsx','xl/workbook.xml','application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml'),
    ('.pptx','ppt/presentation.xml','application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml')])
def test_office_policy_validates_package_metadata(extension,primary,content_type):
    import io,zipfile
    data=io.BytesIO()
    with zipfile.ZipFile(data,'w') as package:
        package.writestr(primary,'<document/>')
        package.writestr('[Content_Types].xml','<Types><Override PartName="/'+primary+'" ContentType="'+content_type+'"/></Types>')
    assert checked_mime('report'+extension,data.getvalue()).startswith('application/')
    with pytest.raises(ArtifactError):checked_mime('renamed'+extension,b'PK\x03\x04arbitrary archive')


def test_recovery_reclaims_proven_dead_holder_after_new_grace(runtime,monkeypatch):
    import time
    agent,db,_=runtime
    holder='pid=999999999:gone'
    assert db.try_acquire_session_turn_lease('source',holder)
    agent._active_session_turn_lease_holder=holder
    artifact=capture(runtime)
    monkeypatch.setattr('hermes_state._compression_lock_holder_process_is_dead',lambda candidate:candidate==holder)
    now=time.time()
    assert db.recover_artifact_captures(now=now)==0
    assert db._conn.execute('SELECT COUNT(*) FROM session_turn_leases').fetchone()[0]==0
    assert db._conn.execute('SELECT state FROM pantheon_artifact_turns').fetchone()[0]=='abandoned'
    assert db.artifact_directory.joinpath(artifact+'.blob').exists()
    assert db.recover_artifact_captures(now=now+25*3600)==1


def test_default_output_root_under_realistic_dot_hermes_profile_is_allowed(runtime):
    from agent.pantheon_artifacts import prepare_turn_output
    agent,db,workspace=runtime
    agent._pantheon_profile_home=str(workspace/'.hermes')
    agent._pantheon_artifact_roots=()
    prepare_turn_output(agent)
    source=Path(agent._pantheon_turn_output_root)/'output.txt';source.write_text('result')
    result=dispatch(agent,'default','task-a',{'path':str(source)},native_environment=SimpleNamespace(cwd=str(workspace)))
    assert json.loads(result)['status']=='captured',result
