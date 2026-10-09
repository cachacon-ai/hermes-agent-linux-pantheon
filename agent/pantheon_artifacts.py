"""Session-gated explicit publication and common response-commit integration."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import hashlib
import json
import logging
import os
from pathlib import Path
import stat
import tempfile
import uuid

from hermes_artifacts import ArtifactError, MAX_ARTIFACT_BYTES, capture_lock

TOOL_NAME = 'pantheon_publish_artifact'
logger = logging.getLogger(__name__)
_INVOCATION = ContextVar('pantheon_artifact_invocation', default=None)

SCHEMA = {
    'name': TOOL_NAME,
    'description': ('Explicitly publish a completed deliverable to Pantheon. Captures an immutable file snapshot now; delivery waits for the successful final assistant response. Only files inside this session workspace are accepted. No MEDIA tags or prose publish files. Maximum 25 MiB. Self-contained HTML, raster images, and supported documents.'),
    'parameters': {'type':'object','properties': {
        'path':{'type':'string','description':'Absolute source path in the producing workspace.'},
        'name':{'type':'string','description':'Optional display basename; never changes ownership or content type.'}},
        'required':['path'],'additionalProperties':False},
}

@dataclass(frozen=True)
class PublicationInvocation:
    db: object
    profile: str
    root: str
    session_id: str
    turn_id: str
    call_id: str
    nonce: str
    environment: object
    task_id: str
    native: bool
    roots: tuple[str,...]


@dataclass(frozen=True)
class TurnAuthority:
    db: object
    profile: str
    session_id: str
    turn_id: str
    holder: object
    environment: object
    roots: tuple[str,...]


def freeze_authority(agent, task_id, *, native_environment=None):
    """Coordinator freezes provenance BEFORE scheduling any worker."""
    if not enabled(agent): return None
    if native_environment is None:
        from tools.terminal_tool import get_active_env
        environment=get_active_env(task_id)
    else:
        environment=native_environment
    try:
        roots=_source_roots(agent,environment)
    except ArtifactError:
        roots=()
    return TurnAuthority(agent._session_db,str(agent._pantheon_profile),agent.session_id,
                         agent._current_turn_id,getattr(agent,'_active_session_turn_lease_holder',None),
                         environment,roots)


def enabled(agent):
    return getattr(agent, '_pantheon_artifact_delivery_version', 0) == 1


def is_cancelled(agent, turn_id=None):
    turn_id = turn_id or getattr(agent, '_current_turn_id', None)
    return (getattr(agent, '_pantheon_cancelled_turn_id', None) == turn_id or
            (turn_id == getattr(agent, '_current_turn_id', None) and
             getattr(agent, '_interrupt_requested', False)))


def safe_source_bytes(path, roots, *, limit=MAX_ARTIFACT_BYTES):
    """Open every component once, no symlink traversal or special-file blocking."""
    if os.name != 'posix' or not hasattr(os,'O_NOFOLLOW'):
        raise ArtifactError('unsupported_source','POSIX no-follow source reads are required')
    candidate=Path(path)
    if not candidate.is_absolute() or '..' in candidate.parts or '\x00' in str(candidate):
        raise ArtifactError('invalid_source','Source must be an absolute root-contained path')
    approved=None
    for raw_root in roots:
        root=Path(raw_root)
        try:
            relative=candidate.relative_to(root)
        except ValueError:
            continue
        if not relative.parts:
            continue
        approved=(root,relative)
        break
    if approved is None:
        raise ArtifactError('source_outside_root','Source is outside approved workspace roots')
    root,relative=approved
    forbidden={'.git','.ssh','.aws','.gnupg','.codex','.hermes','pantheon-artifacts'}
    if any(part in forbidden or part == '.env' or part.startswith('.env.') for part in relative.parts):
        raise ArtifactError('protected_source','Protected configuration files cannot be published')
    fds=[]
    try:
        # Start at / so even a symlink in the trusted root path is rejected.
        directory=os.open('/',os.O_RDONLY|os.O_DIRECTORY)
        fds.append(directory)
        for component in root.parts[1:]+relative.parts[:-1]:
            directory=os.open(component,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=directory)
            fds.append(directory)
        fd=os.open(relative.name,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK,dir_fd=directory)
        fds.append(fd)
        before=os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise ArtifactError('unsupported_source','Source must be a regular file')
        if before.st_size>limit:
            raise ArtifactError('file_too_large','File exceeds 25 MiB')
        chunks=[]; length=0
        while True:
            chunk=os.read(fd,min(1024*1024,limit+1-length))
            if not chunk: break
            chunks.append(chunk);length+=len(chunk)
            if length>limit: raise ArtifactError('file_too_large','File exceeds 25 MiB')
        data=b''.join(chunks)
        after=os.fstat(fd)
        fingerprint=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
        if fingerprint(before)!=fingerprint(after) or length!=after.st_size:
            raise ArtifactError('source_changed','Source changed during capture; retry explicitly')
        # Same-size in-place rewrites may leave timestamps unchanged on coarse
        # filesystems; confirm the path still exposes the bytes we captured.
        verify_fd=os.open(relative.name,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK,dir_fd=directory)
        try:
            verify_stat=os.fstat(verify_fd)
            if (verify_stat.st_dev,verify_stat.st_ino)!=(before.st_dev,before.st_ino):
                raise ArtifactError('source_changed','Source changed during capture; retry explicitly')
            verify=bytearray(); offset=0
            while offset<length:
                chunk=os.pread(verify_fd,min(1024*1024,length-offset),offset)
                if not chunk: break
                verify.extend(chunk); offset+=len(chunk)
            if bytes(verify)!=data or len(verify)!=length:
                raise ArtifactError('source_changed','Source changed during capture; retry explicitly')
        finally:
            os.close(verify_fd)
        return data
    except OSError as exc:
        raise ArtifactError('source_unavailable','Source could not be opened safely') from exc
    finally:
        for fd in reversed(fds): os.close(fd)


def _valid_office_package(data, suffix):
    """Validate bounded package metadata without extracting document content."""
    import io
    import zipfile
    import xml.etree.ElementTree as ET
    primary={'.docx':'word/document.xml','.xlsx':'xl/workbook.xml','.pptx':'ppt/presentation.xml'}[suffix]
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as package:
            info=package.getinfo('[Content_Types].xml')
            if info.file_size>65536 or info.flag_bits & 1 or primary not in package.namelist(): return False
            with package.open(info) as source:
                metadata=source.read(65537)
            if len(metadata)>65536: return False
            root=ET.fromstring(metadata)
            # OOXML package main content types differ from the file MIME.
            expected={'.docx':'application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml',
                      '.xlsx':'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml',
                      '.pptx':'application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml'}[suffix]
            return any(node.attrib.get('PartName')=='/'+primary and node.attrib.get('ContentType')==expected for node in root)
    except (OSError,ValueError,KeyError,zipfile.BadZipFile,RuntimeError,ET.ParseError):
        return False


def checked_mime(source_name, data):
    suffix=Path(source_name).suffix.lower()
    policies={'.png':('image/png',lambda b:b.startswith(b'\x89PNG\r\n\x1a\n')),
              '.jpg':('image/jpeg',lambda b:b.startswith(b'\xff\xd8\xff')),
              '.jpeg':('image/jpeg',lambda b:b.startswith(b'\xff\xd8\xff')),
              '.gif':('image/gif',lambda b:b[:6] in (b'GIF87a',b'GIF89a')),
              '.webp':('image/webp',lambda b:b.startswith(b'RIFF') and b[8:12]==b'WEBP'),
              '.pdf':('application/pdf',lambda b:b.startswith(b'%PDF-')),
              '.docx':('application/vnd.openxmlformats-officedocument.wordprocessingml.document',lambda b:b.startswith(b'PK\x03\x04')),
              '.xlsx':('application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',lambda b:b.startswith(b'PK\x03\x04')),
              '.pptx':('application/vnd.openxmlformats-officedocument.presentationml.presentation',lambda b:b.startswith(b'PK\x03\x04'))}
    if suffix in policies:
        mime,check=policies[suffix]
        if not check(data[:32]) or suffix in {'.docx','.xlsx','.pptx'} and not _valid_office_package(data,suffix):
            raise ArtifactError('type_mismatch','File bytes do not match its extension')
        return mime
    if suffix in {'.html','.htm','.txt','.md','.csv','.json','.rtf'}:
        try: text=data.decode('utf-8')
        except UnicodeDecodeError as exc: raise ArtifactError('type_mismatch','Text deliverables must be UTF-8') from exc
        if '\x00' in text: raise ArtifactError('type_mismatch','Text contains binary content')
        if suffix in {'.html','.htm'}:
            if not any(marker in text[:4096].lower() for marker in ('<!doctype html','<html','<body','<head')):
                raise ArtifactError('type_mismatch','HTML source needs an HTML document marker')
            return 'text/html'
        if suffix=='.rtf' and not text.lstrip().startswith('{\\rtf'):
            raise ArtifactError('type_mismatch','RTF source needs a document header')
        if suffix=='.json':
            try: json.loads(text)
            except ValueError as exc: raise ArtifactError('type_mismatch','JSON source is invalid') from exc
        return {'.txt':'text/plain','.md':'text/markdown','.csv':'text/csv','.json':'application/json','.rtf':'application/rtf'}[suffix]
    raise ArtifactError('unsupported_type','Unsupported or active file type; publish a supported document')


def _source_roots(agent, environment):
    configured=tuple(getattr(agent,'_pantheon_artifact_roots',()))
    output=getattr(agent,'_pantheon_turn_output_root',None)
    if output:
        configured=(output,*configured)
    roots=[]
    for raw in configured:
        p=Path(raw)
        if not p.is_absolute() or p in (Path('/'),Path.home(),Path(getattr(agent,'_pantheon_profile_home','/nonexistent'))) or (raw != output and any(part in {'.git','.ssh','.aws','.gnupg','pantheon-artifacts'} for part in p.parts)):
            continue
        # A generic env can move within the approved workspace. Never expand
        # authority to a new cwd merely because a tool changed directories.
        roots.append(str(p))
    if not roots:
        raise ArtifactError('unsupported_source','Session has no approved project workspace')
    return tuple(roots)


@contextmanager
def invocation(agent, call_id, task_id, *, native_environment=None, authority=None):
    if not enabled(agent):
        raise ArtifactError('capability_unavailable','This session does not support publication')
    authority=authority or freeze_authority(agent,task_id,native_environment=native_environment)
    db=authority.db if authority else None
    if db is None: raise ArtifactError('store_unavailable','Durable publication store is unavailable')
    environment=authority.environment
    if native_environment is None:
        from tools.environments.local import LocalEnvironment
        if not isinstance(environment,LocalEnvironment):
            raise ArtifactError('unsupported_source','Publication requires an existing POSIX local producing environment; this backend has no safe-reader adapter')
    roots=authority.roots
    if not roots: raise ArtifactError('unsupported_source','No approved producing output root')
    session_id=authority.session_id;turn_id=authority.turn_id
    if is_cancelled(agent, turn_id):
        raise ArtifactError('invocation_revoked', 'Publication turn was cancelled')
    root,nonce=db.artifact_begin_call(session_id=session_id,turn_id=turn_id,call_id=call_id,holder=authority.holder)
    context=PublicationInvocation(db,authority.profile,root,session_id,turn_id,call_id,nonce,environment,task_id,native_environment is not None,roots)
    token=_INVOCATION.set(context)
    try: yield context
    finally: _INVOCATION.reset(token)


def publish(args):
    context=_INVOCATION.get()
    if context is None:
        return json.dumps({'error':'missing_context','message':'No trusted active publication invocation'})
    snapshot=None
    try:
        if not isinstance(args,dict) or set(args)-{'path','name'} or not isinstance(args.get('path'),str):
            raise ArtifactError('invalid_arguments','Expected path and optional display name only')
        with context.db._read_ctx() as conn:
            existing=conn.execute("""SELECT a.state,a.descriptor FROM pantheon_artifact_calls c
                JOIN pantheon_artifacts a ON a.artifact_id=c.artifact_id
                WHERE c.root=? AND c.turn_id=? AND c.call_id=? AND c.nonce=?
                  AND c.state IN ('open','accepted')""",(context.root,context.turn_id,context.call_id,context.nonce)).fetchone()
        if existing and existing['state'] in {'staged','ready'}:
            captured=json.loads(existing['descriptor'])
            return json.dumps({'status':'captured','delivered':False,'pending_response_commit':True,
                               'artifact_id':captured['artifact_id'],'name':captured['name'],'size':captured['size']})
        if existing:
            raise ArtifactError('artifact_deleted','This publication was removed')
        if not context.native:
            from tools.terminal_tool import get_active_env
            if get_active_env(context.task_id) is not context.environment:
                raise ArtifactError('source_changed','Producing environment is no longer available')
        source=args['path']; data=safe_source_bytes(source,context.roots)
        if not context.native and get_active_env(context.task_id) is not context.environment:
            raise ArtifactError('source_changed','Producing environment changed during capture')
        mime=checked_mime(source,data)
        name=args.get('name') or Path(source).name
        if not isinstance(name,str) or not name.strip() or len(name)>255 or '/' in name or '\\' in name or name in {'.','..'} or any(ord(c)<32 for c in name):
            raise ArtifactError('invalid_name','Display name must be a safe basename')
        if Path(name).suffix.lower()!=Path(source).suffix.lower():
            raise ArtifactError('type_mismatch','Display name must preserve the source extension')
        directory=context.db.artifact_directory
        directory.mkdir(mode=0o700,parents=True,exist_ok=True)
        if directory.is_symlink(): raise ArtifactError('snapshot_unavailable','Snapshot directory must not be a symlink')
        with capture_lock(directory):
            artifact_id=str(uuid.uuid4()); snapshot=artifact_id+'.blob'
            fd,tmp=tempfile.mkstemp(prefix='.capture-',dir=directory)
            try:
                with os.fdopen(fd,'wb') as out:
                    out.write(data);out.flush();os.fsync(out.fileno())
                os.replace(tmp,directory/snapshot)
                dirfd=os.open(directory,os.O_RDONLY|os.O_DIRECTORY)
                try: os.fsync(dirfd)
                finally: os.close(dirfd)
            finally:
                if os.path.exists(tmp): os.unlink(tmp)
            import datetime
            descriptor=dict(schema_version=1,artifact_id=artifact_id,profile=context.profile,conversation_root_id=context.root,
                            source_session_id=context.session_id,source_turn_id=context.turn_id,
                            source_delivery_id=context.turn_id,source_row_id=None,name=name,mime=mime,size=len(data),
                            sha256=hashlib.sha256(data).hexdigest(),created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                            source_visible=True,producer={'profile':context.profile})
            staged=context.db.artifact_stage(root=context.root,session_id=context.session_id,turn_id=context.turn_id,call_id=context.call_id,nonce=context.nonce,descriptor=descriptor,snapshot=snapshot)
            if staged['artifact_id']!=artifact_id: (directory/snapshot).unlink(missing_ok=True)
        return json.dumps({'status':'captured','delivered':False,'pending_response_commit':True,'artifact_id':staged['artifact_id'],'name':staged['name'],'size':staged['size']})
    except (ArtifactError,OSError) as exc:
        if snapshot:
            (context.db.artifact_directory/snapshot).unlink(missing_ok=True)
        return json.dumps({'error':getattr(exc,'code','snapshot_unavailable'),'message':str(exc)})


def dispatch(agent,call_id,task_id,args,*,native_environment=None,authority=None):
    try:
        with invocation(agent,call_id,task_id,native_environment=native_environment,authority=authority):
            return publish(args)
    except ArtifactError as exc:
        return json.dumps({'error':exc.code,'message':str(exc)})


def accept_result(agent, call_id, result, *, abandoned=False, authority=None):
    if not enabled(agent): return
    try:
        try:
            value=json.loads(result) if isinstance(result,str) else {}
        except (ValueError,TypeError):
            value={}
        if not isinstance(value,dict): value={}
        if authority is None:
            db,session_id,turn_id=agent._session_db,agent.session_id,agent._current_turn_id
        else:
            db,session_id,turn_id=authority.db,authority.session_id,authority.turn_id
        accepted=not abandoned and not is_cancelled(agent,turn_id) and value.get('status')=='captured' and value.get('pending_response_commit') is True
        db.artifact_finish_call(session_id,turn_id,call_id,accepted)
    except ArtifactError:
        if not abandoned: raise


def cancel_turn(agent):
    if enabled(agent) and isinstance(getattr(agent,'_current_turn_id',None),str):
        # This process's turn fence survives a transient database failure and
        # a later clear_interrupt() within the same generation.
        agent._pantheon_cancelled_turn_id = agent._current_turn_id
        try:
            agent._session_db.artifact_cancel_turn(agent.session_id,agent._current_turn_id)
        except Exception as exc:
            logger.warning('Artifact cancellation persistence failed (%s)', type(exc).__name__)
            return False
        return True


def commit_response(agent,messages,history=None):
    """Select the exact terminal row and make its publication atomic with flush."""
    if not enabled(agent):
        return agent._flush_messages_to_session_db(messages,history)
    if is_cancelled(agent):
        cancel_turn(agent)
        return False
    db=agent._session_db
    with db._read_ctx() as conn:
        root=db._artifact_root_on_conn(conn,agent.session_id)
        turn=conn.execute('SELECT state FROM pantheon_artifact_turns WHERE root=? AND turn_id=?',(root,agent._current_turn_id)).fetchone()
    if not turn:
        return agent._flush_messages_to_session_db(messages,history)
    tail=messages[-1] if messages else None
    if not isinstance(tail,dict) or tail.get('role')!='assistant' or tail.get('tool_calls'):
        from agent.message_metadata import append_message
        tail=append_message(messages,{'role':'assistant','content':''})
    finalization=dict(turn_id=agent._current_turn_id,generation=agent._current_turn_id,
                      holder=getattr(agent,'_active_session_turn_lease_holder',None),message=tail)
    agent._pantheon_pending_commit=finalization
    try:
        ok=agent._flush_messages_to_session_db(messages,history)
        if ok is False: return False
        if 'committed_metadata' in finalization:
            tail['display_metadata']=finalization['committed_metadata']
        agent._pantheon_committed_artifacts=finalization.get('artifacts',[])
        agent._pantheon_reply_row_id=tail.get('_row_id')
        return ok
    finally:
        agent._pantheon_pending_commit=None


def reply_presentation(message):
    """Presentation fallback remains outside provider-visible content."""
    content=message.get('content') or ''
    if isinstance(content,str) and content.strip(): return content
    meta=message.get('display_metadata')
    return meta.get('pantheon_reply_preview','') if isinstance(meta,dict) else ''


def prepare_turn_output(agent):
    """Allocate the turn's output root before tools, without a cached-prefix edit.

    Local terminal execution and native Codex share this host. Remote terminal
    backends are explicitly unavailable until they supply a safe-reader adapter;
    we never create a replacement environment to make publication appear to work.
    """
    if not enabled(agent): return ''
    from tools.terminal_scope import terminal_env
    native = getattr(agent,'api_mode',None)=='codex_app_server'
    if not native and terminal_env('TERMINAL_ENV','local') != 'local':
        agent._pantheon_turn_output_root=None
        return '[Pantheon publication: this producing environment has no supported safe-reader adapter; file publication is unavailable.]'
    base=Path(agent._pantheon_profile_home).resolve()/'pantheon-output'
    session=hashlib.sha256(str(agent.session_id).encode()).hexdigest()[:24]
    turn=hashlib.sha256(str(agent._current_turn_id).encode()).hexdigest()[:24]
    root=base/session/turn
    root.mkdir(mode=0o700,parents=True,exist_ok=True)
    for path in (base,base/session,root):
        if path.is_symlink():
            raise ArtifactError('unsupported_source','Output location must not contain symlinks')
    agent._pantheon_turn_output_root=str(root)
    return ('[Pantheon file delivery: write completed deliverables for this turn in '+str(root)+
            '. Call pantheon_publish_artifact with the absolute path explicitly. Captured bytes are pending until the final assistant response commits. Do not use MEDIA tags. Maximum 25 MiB.]')


def has_accepted_publication(agent):
    """Only an admitted captured invocation can make an empty stop terminal."""
    if not enabled(agent) or is_cancelled(agent): return False
    db=getattr(agent,'_session_db',None)
    if db is None: return False
    with db._read_ctx() as conn:
        root=db._artifact_root_on_conn(conn,agent.session_id)
        return conn.execute("""SELECT 1 FROM pantheon_artifacts a
            JOIN pantheon_artifact_calls c ON c.root=a.root AND c.turn_id=a.turn_id AND c.call_id=a.call_id
            JOIN pantheon_artifact_turns t ON t.root=a.root AND t.turn_id=a.turn_id
            WHERE a.root=? AND a.turn_id=? AND a.state='staged'
              AND c.state='accepted' AND t.state='active' LIMIT 1""",(root,agent._current_turn_id)).fetchone() is not None


def capability_details(profile_home=None):
    """Preflight the actual producer and native protocol without a model call."""
    from hermes_constants import get_hermes_home
    import yaml
    def unavailable(code, reason):
        return {"artifact_delivery_version": 0,
                "artifact_delivery_unavailable_code": code,
                "artifact_delivery_unavailable_reason": reason}
    home=Path(profile_home or get_hermes_home())
    path=home/'config.yaml'
    try:
        cfg=yaml.safe_load(path.read_text()) if path.exists() else {}
        cfg=cfg or {}
        if not isinstance(cfg,dict):
            return unavailable('invalid_configuration','The producing profile configuration is invalid.')
        if os.name!='posix' or not hasattr(os,'O_NOFOLLOW'):
            return unavailable('unsupported_source','This host does not support safe POSIX file capture.')
        model=cfg.get('model',{})
        model=model if isinstance(model,dict) else {}
        native=(model.get('openai_runtime',cfg.get('openai_runtime'))=='codex_app_server'
                or model.get('api_mode',cfg.get('api_mode'))=='codex_app_server')
        if native:
            from agent.transports.codex_app_server_session import supports_pantheon_dynamic_tools
            if not supports_pantheon_dynamic_tools():
                return unavailable('unsupported_protocol','The installed Codex adapter does not support the required publication protocol.')
        else:
            terminal=cfg.get('terminal',{})
            terminal=terminal if isinstance(terminal,dict) else {}
            backend=str(os.environ.get('TERMINAL_ENV') or terminal.get('backend') or 'local').strip().lower()
            if backend!='local':
                return unavailable('unsupported_source','The producing terminal backend has no safe file-capture adapter.')
        return {"artifact_delivery_version": 1}
    except (OSError,ValueError,yaml.YAMLError):
        return unavailable('invalid_configuration','The producing profile configuration could not be read.')


def negotiate_capability(profile_home=None):
    return capability_details(profile_home)['artifact_delivery_version']
