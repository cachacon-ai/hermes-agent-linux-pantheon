"""Explicit publication is authorized by actual executor admission/receipt."""
import json
import threading
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.tool_executor import execute_tool_calls_sequential, execute_tool_calls_concurrent
from agent.pantheon_artifacts import TOOL_NAME, commit_response
from tools.environments.local import LocalEnvironment
from tests.run_agent.test_sequential_tool_timeout import _make_agent
from tests.agent.test_pantheon_artifacts import runtime


def call(call_id,path):
    return SimpleNamespace(id=call_id,type='function',function=SimpleNamespace(name=TOOL_NAME,arguments=json.dumps({'path':str(path)})))


def bind_agent(runtime,tmp_path):
    minimal,db,workspace=runtime
    agent=_make_agent(tmp_path)
    agent._flush_messages_to_session_db=type(minimal)._flush_messages_to_session_db.__get__(agent)
    for attr in ('_session_db','_session_db_created','_last_flushed_db_idx','session_id','_current_turn_id','_current_task_id','_pantheon_artifact_delivery_version','_pantheon_profile','_pantheon_profile_home','_pantheon_artifact_roots','_active_session_turn_lease_holder'):
        setattr(agent,attr,getattr(minimal,attr))
    agent.valid_tool_names={TOOL_NAME}
    environment=LocalEnvironment.__new__(LocalEnvironment);environment.cwd=str(workspace)
    return agent,db,workspace,environment


@pytest.mark.parametrize('executor',[execute_tool_calls_sequential,execute_tool_calls_concurrent])
def test_actual_executor_dispatch_receipt_and_final_binding(runtime,tmp_path,monkeypatch,executor):
    agent,db,workspace,environment=bind_agent(runtime,tmp_path)
    path=workspace/'report.txt';path.write_text('actual executor')
    monkeypatch.setattr('tools.terminal_tool.get_active_env',lambda task:environment)
    calls=[call('one',path),call('two',path)]
    messages=[{'role':'user','content':'generate','timestamp':1780000000.0},
              {'role':'assistant','content':None,'tool_calls':[{'id':c.id,'type':'function','function':{'name':TOOL_NAME,'arguments':c.function.arguments}} for c in calls],'timestamp':1780000001.0}]
    executor(agent,SimpleNamespace(tool_calls=calls),messages,'task-a')
    results=[json.loads(row['content']) for row in messages if row['role']=='tool']
    assert len(results)==2
    assert all(r.get('status')=='captured' and r['delivered'] is False for r in results),results
    assert db.artifact_changes('source','alpha')['changes']==[]
    messages.append({'role':'assistant','content':'Done','timestamp':1780000002.0})
    assert commit_response(agent,messages)
    changes=db.artifact_changes('source','alpha')['changes']
    assert len(changes)==2
    assert len({change['artifact']['artifact_id'] for change in changes})==2
    assert len({change['artifact']['source_row_id'] for change in changes})==1


@pytest.mark.parametrize('executor',[execute_tool_calls_sequential,execute_tool_calls_concurrent])
def test_late_abandoned_worker_cannot_authorize_publication(runtime,tmp_path,monkeypatch,executor):
    agent,db,workspace,environment=bind_agent(runtime,tmp_path)
    path=workspace/'late.txt';path.write_text('late')
    monkeypatch.setattr('tools.terminal_tool.get_active_env',lambda task:environment)
    monkeypatch.setattr('agent.tool_executor._resolve_concurrent_tool_timeout',lambda:0.2)
    monkeypatch.setattr('agent.tool_executor._resolve_sequential_tool_timeout',lambda:0.2)
    entered=threading.Event();release=threading.Event();finished=threading.Event()
    from agent import pantheon_artifacts
    real_read=pantheon_artifacts.safe_source_bytes
    def slow_read(*args,**kwargs):
        entered.set();release.wait(timeout=10)
        try:return real_read(*args,**kwargs)
        finally:finished.set()
    monkeypatch.setattr(pantheon_artifacts,'safe_source_bytes',slow_read)
    calls=[call('late',path)]
    messages=[]
    try:
        executor(agent,SimpleNamespace(tool_calls=calls),messages,'task-a')
        assert entered.is_set()
        release.set();assert finished.wait(timeout=5)
        messages.append({'role':'assistant','content':'Done','timestamp':1780000002.0})
        commit_response(agent,messages)
        assert db.artifact_changes('source','alpha')['changes']==[]
    finally:release.set()


def test_attachment_only_stop_finishes_without_empty_response_retry(runtime,tmp_path,monkeypatch):
    from unittest.mock import MagicMock
    from tests.run_agent.test_run_agent import _mock_response
    agent,db,workspace,environment=bind_agent(runtime,tmp_path)
    path=workspace/'report.txt';path.write_text('file only')
    monkeypatch.setattr('tools.terminal_tool.get_active_env',lambda task:environment)
    agent.client=MagicMock()
    agent.client.chat.completions.create.side_effect=[
        _mock_response(content=None,finish_reason='tool_calls',tool_calls=[call('publish',path)]),
        _mock_response(content='',finish_reason='stop'),
    ]
    agent._cached_system_prompt='You are helpful.'
    agent._use_prompt_caching=False
    agent.compression_enabled=False
    agent.save_trajectories=False
    agent.tool_delay=0
    with patch.object(agent,'_cleanup_task_resources'),patch.object(agent,'_save_trajectory'):
        result=agent.run_conversation('Generate a deliverable',task_id='task-a')
    assert not result.get('failed'),result
    assert result['api_calls']==2
    assert agent.client.chat.completions.create.call_count==2
    assert result['final_response']==''
    assert len(result['artifacts'])==1
    source=db.get_messages_as_conversation('source',include_row_ids=True)[-1]
    assert source['content']==''
    assert source['display_metadata']['pantheon_reply_preview']=='Generated report.txt.'


def test_concurrent_timeout_revokes_before_late_success_grace(runtime,tmp_path,monkeypatch):
    from agent import pantheon_artifacts
    import tools.daemon_pool as daemon_pool
    agent,db,workspace,environment=bind_agent(runtime,tmp_path)
    path=workspace/'late.txt';path.write_text('captured before timeout')
    monkeypatch.setattr('tools.terminal_tool.get_active_env',lambda task:environment)
    monkeypatch.setattr('agent.tool_executor._resolve_concurrent_tool_timeout',lambda:0.2)
    captured=threading.Event();release=threading.Event();futures=[]
    original_publish=pantheon_artifacts.publish
    def delayed_receipt(args):
        result=original_publish(args)
        assert json.loads(result)['status']=='captured'
        captured.set();release.wait(timeout=10)
        return result
    monkeypatch.setattr(pantheon_artifacts,'publish',delayed_receipt)
    original_executor=daemon_pool.DaemonThreadPoolExecutor
    class RecordingExecutor(original_executor):
        def submit(self,*args,**kwargs):
            future=super().submit(*args,**kwargs);futures.append(future)
            assert captured.wait(timeout=5)
            return future
    monkeypatch.setattr(daemon_pool,'DaemonThreadPoolExecutor',RecordingExecutor)
    def interrupt(flag,*args):
        if flag:
            release.set()
            futures[0].result(timeout=5)
    monkeypatch.setattr('run_agent._set_interrupt',interrupt)
    messages=[]
    try:
        execute_tool_calls_concurrent(agent,SimpleNamespace(tool_calls=[call('late',path)]),messages,'task-a')
        assert 'timed out' in messages[-1]['content']
        messages.append({'role':'assistant','content':'Text still completed','timestamp':1780000002.0})
        assert commit_response(agent,messages)
        assert db.artifact_changes('source','alpha')['changes']==[]
        assert db.cleanup_abandoned_artifacts(now=10**11)==1
    finally:release.set()
