"""Native Codex dynamic publication cannot borrow child/stale contexts."""
from agent.transports.codex_app_server_session import CodexAppServerSession
from tests.agent.transports.test_codex_app_server_session import FakeClient


def test_dynamic_tool_capability_and_exact_parent_scope(monkeypatch):
    monkeypatch.setattr("agent.transports.codex_app_server_session.supports_pantheon_dynamic_tools",lambda *_:True)
    handled=[]
    session=CodexAppServerSession(cwd='/workspace',client_factory=FakeClient,
        dynamic_tools=[{'name':'pantheon_publish_artifact','description':'publish','inputSchema':{'type':'object'}}],
        dynamic_tool_handler=lambda params:handled.append(params) or {'success':True,'contentItems':[]})
    thread=session.ensure_started();session._active_turn_id='current'
    assert session._client.requests[0][1]['dynamicTools'][0]['name']=='pantheon_publish_artifact'
    base={'threadId':thread,'turnId':'current','callId':'call-1','tool':'pantheon_publish_artifact','arguments':{'path':'/workspace/report.html'}}
    for changes in ({'threadId':'child'},{'turnId':'stale'},{'callId':''},{'tool':'other'},{}):
        params={**base,**changes}
        session._handle_server_request({'id':len(handled),'method':'item/tool/call','params':params})
    assert handled==[base]
    assert [response[1]['success'] for response in session._client.responses]==[False,False,False,False,True]
    session._interrupt_event.set()
    session._handle_server_request({'id':9,'method':'item/tool/call','params':base})
    assert len(handled)==1 and session._client.responses[-1][1]['success'] is False
    session.close()


def test_unsupported_native_protocol_retains_text_runtime(monkeypatch):
    monkeypatch.setattr("agent.transports.codex_app_server_session.supports_pantheon_dynamic_tools",lambda *_:False)
    session=CodexAppServerSession(cwd="/workspace",client_factory=FakeClient,
        dynamic_tools=[{"name":"pantheon_publish_artifact","description":"publish","inputSchema":{"type":"object"}}],
        dynamic_tool_handler=lambda params: {"success":True})
    assert session.artifact_delivery_available is False
    session.ensure_started()
    assert "dynamicTools" not in session._client.requests[0][1]
    session.close()
