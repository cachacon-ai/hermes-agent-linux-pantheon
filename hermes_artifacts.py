"""Durable, profile-local Pantheon publication registry.

All visibility transitions share the transcript writer transaction. Files are
private immutable snapshots; paths never appear in descriptors or changefeed.
"""
from __future__ import annotations

import base64
import atexit
from contextlib import contextmanager
import hashlib
import hmac
import json
import logging
import os
import re
import stat
import threading
from pathlib import Path
import time
import uuid

MAX_ARTIFACT_BYTES = 25 * 1024 * 1024
logger = logging.getLogger(__name__)


class ArtifactError(ValueError):
    def __init__(self, code, message, status=400):
        super().__init__(message)
        self.code, self.status = code, status


_ARTIFACT_SCHEMA_STATEMENTS = (
    """CREATE TABLE IF NOT EXISTS pantheon_artifact_turns (
        root TEXT NOT NULL, turn_id TEXT NOT NULL, session_id TEXT NOT NULL,
        generation TEXT NOT NULL, holder TEXT, state TEXT NOT NULL,
        created_at REAL NOT NULL, ended_at REAL,
        PRIMARY KEY(root, turn_id))""",
    """CREATE TABLE IF NOT EXISTS pantheon_artifact_calls (
        root TEXT NOT NULL, turn_id TEXT NOT NULL, call_id TEXT NOT NULL,
        nonce TEXT NOT NULL, state TEXT NOT NULL, artifact_id TEXT,
        PRIMARY KEY(root,turn_id,call_id))""",
    """CREATE TABLE IF NOT EXISTS pantheon_artifacts (
        artifact_id TEXT PRIMARY KEY, root TEXT NOT NULL, turn_id TEXT,
        call_id TEXT, session_id TEXT, state TEXT NOT NULL, descriptor TEXT,
        snapshot TEXT, source_row_id INTEGER, source_physical_row INTEGER,
        created_at REAL NOT NULL, abandoned_at REAL,
        UNIQUE(root,turn_id,call_id))""",
    """CREATE TABLE IF NOT EXISTS pantheon_artifact_changes (
        revision INTEGER PRIMARY KEY AUTOINCREMENT, root TEXT NOT NULL,
        artifact_id TEXT NOT NULL, type TEXT NOT NULL, payload TEXT NOT NULL)""",
    "CREATE TABLE IF NOT EXISTS pantheon_artifact_cleanup (snapshot TEXT PRIMARY KEY, root TEXT NOT NULL)",
    "CREATE TABLE IF NOT EXISTS pantheon_artifact_orphans (snapshot TEXT PRIMARY KEY, abandoned_at REAL NOT NULL)",
    "CREATE TABLE IF NOT EXISTS pantheon_artifact_segments (session_id TEXT PRIMARY KEY, root TEXT NOT NULL)",
    "CREATE INDEX IF NOT EXISTS pantheon_artifact_segment_root ON pantheon_artifact_segments(root)",
    "CREATE INDEX IF NOT EXISTS pantheon_artifact_feed ON pantheon_artifact_changes(root,revision)",
    """CREATE TABLE IF NOT EXISTS pantheon_artifact_owners (
        root TEXT PRIMARY KEY, erased INTEGER NOT NULL DEFAULT 0,
        cursor_key TEXT NOT NULL)""",
)
ARTIFACT_SCHEMA_SQL = ";\n".join(_ARTIFACT_SCHEMA_STATEMENTS) + ";\n"


def recover_profile_artifact_captures(launch_db=None):
    """Background-only recovery for the shared RPC backend's local catalog.

    Catalog enumeration is lightweight and never starts a provider or agent.
    Existing dedicated databases are closed here; a supplied launch handle
    remains owned by its caller. SQLite mode=rw refuses a concurrently removed
    database instead of bootstrapping it. One failed profile cannot block the
    others; each database's holder CAS and capture lock remain authoritative.
    """
    from hermes_cli.profiles import profiles_to_serve
    from hermes_constants import get_default_hermes_root, named_profile_is_deleted
    from hermes_state import SessionDB
    results = {}
    seen = set()
    if launch_db is not None:
        seen.add(Path(launch_db.db_path).resolve())
        try:
            results[str(launch_db.db_path)] = launch_db.recover_artifact_captures()
        except Exception as exc:
            logger.warning('Launch profile artifact recovery failed (%s)', type(exc).__name__)
    try:
        profiles = profiles_to_serve(multiplex=True)
        catalog_root = get_default_hermes_root().resolve()
    except Exception as exc:
        logger.warning('Artifact profile catalog unavailable (%s)', type(exc).__name__)
        return results
    for name, home in profiles:
        home = Path(home)
        path = home/'state.db'
        try:
            if (home.is_symlink() or path.is_symlink() or
                    (home/'pantheon-artifacts').is_symlink() or
                    named_profile_is_deleted(home) or not path.is_file()):
                continue
            resolved = path.resolve()
            expected_home = catalog_root if name == 'default' else catalog_root/'profiles'/name
            if home.resolve() != expected_home:
                continue
            if resolved in seen:
                continue
            seen.add(resolved)
            with SessionDB(path, create_if_missing=False) as db:
                if not named_profile_is_deleted(home):
                    results[str(path)] = db.recover_artifact_captures()
        except Exception as exc:
            logger.warning('Profile artifact recovery failed (%s)', type(exc).__name__)
    return results


_housekeeping_lock = threading.Lock()
_housekeeping_thread = None


def _profile_artifact_housekeeping_loop(stop_event):
    while not stop_event.is_set():
        recover_profile_artifact_captures()
        stop_event.wait(3600)


def start_profile_artifact_housekeeping():
    """Once per backend, independent of heartbeat/orphan-sweep opt-outs."""
    global _housekeeping_thread
    with _housekeeping_lock:
        if _housekeeping_thread is not None and _housekeeping_thread.is_alive():
            return False
        stop_event = threading.Event()
        thread = threading.Thread(target=_profile_artifact_housekeeping_loop,
                                  args=(stop_event,), name='pantheon-artifact-housekeeping', daemon=True)
        thread.start()
        _housekeeping_thread = thread
        atexit.register(stop_event.set)
        return True


@contextmanager
def capture_lock(directory, *, exclusive=False, nonblocking=False):
    """Cross-process snapshot admission fence, independent of SQLite commits."""
    import fcntl
    directory=Path(directory)
    directory.mkdir(mode=0o700,parents=True,exist_ok=True)
    if directory.is_symlink():
        raise ArtifactError('snapshot_unavailable','Snapshot directory must not be a symlink')
    fd=os.open(directory/'.capture.lock',os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
    acquired=False
    try:
        mode=fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
        if nonblocking: mode |= fcntl.LOCK_NB
        try:
            fcntl.flock(fd,mode)
            acquired=True
        except BlockingIOError:
            if not nonblocking: raise
        yield acquired
    finally:
        if acquired: fcntl.flock(fd,fcntl.LOCK_UN)
        os.close(fd)


class SessionArtifactsMixin:
    @property
    def artifact_directory(self):
        return Path(self.db_path).parent / "pantheon-artifacts"

    def artifact_conversation_root(self, session_id):
        with self._read_ctx() as conn:
            return self._artifact_root_on_conn(conn, session_id)

    def _artifact_root_on_conn(self, conn, session_id):
        erased = conn.execute("SELECT 1 FROM permanent_session_deletions WHERE session_id=?", (session_id,)).fetchone()
        if erased:
            raise ArtifactError("conversation_deleted", "Conversation was permanently deleted", 410)
        mapped=conn.execute('SELECT root FROM pantheon_artifact_segments WHERE session_id=?',(session_id,)).fetchone()
        root=mapped['root'] if mapped else self._session_turn_lease_key_on_conn(conn,session_id)
        owner=conn.execute('SELECT erased FROM pantheon_artifact_owners WHERE root=?',(root,)).fetchone()
        if owner and owner['erased']:
            raise ArtifactError("conversation_deleted", "Conversation was deleted", 410)
        exists=conn.execute('SELECT 1 FROM sessions WHERE id=?',(session_id,)).fetchone()
        if not exists:
            # Exact historic segment aliases remain valid only while their
            # independently recorded compression conversation still exists.
            survivor=conn.execute('SELECT 1 FROM pantheon_artifact_segments a JOIN sessions s ON s.id=a.session_id WHERE a.root=? LIMIT 1',(root,)).fetchone() if mapped else None
            if not survivor: raise ArtifactError('session_not_found','Unknown session',404)
        return root

    def _artifact_register_segment_on_conn(self, conn, session_id):
        """Persist ownership before compression ancestry can be pruned."""
        row=conn.execute('SELECT * FROM sessions WHERE id=?',(session_id,)).fetchone()
        if row is None: return
        row=dict(row)
        parent_id=row.get('parent_session_id')
        parent=conn.execute('SELECT * FROM sessions WHERE id=?',(parent_id,)).fetchone() if parent_id else None
        def decode_config(raw):
            try: config=json.loads(raw or '{}')
            except (TypeError,ValueError): return {}
            return config if isinstance(config,dict) else {}
        config=decode_config(row['model_config'])
        if parent and parent['end_reason']=='compression' and not self._is_explicit_fork_child_row(row):
            parent_config=decode_config(parent['model_config'])
            if isinstance(parent_config,dict) and parent_config.get('artifact_delivery_version')==1:
                config['artifact_delivery_version']=1
                conn.execute('UPDATE sessions SET model_config=? WHERE id=?',(json.dumps(config),session_id))
        root=self._session_turn_lease_key_on_conn(conn,session_id)
        owner=conn.execute('SELECT 1 FROM pantheon_artifact_owners WHERE root=?',(root,)).fetchone()
        if not owner and config.get('artifact_delivery_version')!=1: return
        conn.execute('INSERT OR IGNORE INTO pantheon_artifact_owners(root,cursor_key) VALUES (?,?)',(root,uuid.uuid4().hex))
        self._artifact_register_family_on_conn(conn,root,session_id)

    def _artifact_register_family_on_conn(self,conn,root,session_id):
        """Backfill existing compression segments without crossing forks."""
        frontier=[root,session_id]
        seen=set()
        while frontier:
            sid=frontier.pop()
            if sid in seen: continue
            seen.add(sid)
            row=conn.execute('SELECT * FROM sessions WHERE id=?',(sid,)).fetchone()
            if row is None: continue
            previous=conn.execute('SELECT root FROM pantheon_artifact_segments WHERE session_id=?',(sid,)).fetchone()
            if previous and previous['root']!=root:
                raise ArtifactError('source_changed','Compression ownership cannot be rebound')
            conn.execute('INSERT OR IGNORE INTO pantheon_artifact_segments VALUES (?,?)',(sid,root))
            if row['end_reason']=='compression':
                for child in conn.execute('SELECT * FROM sessions WHERE parent_session_id=?',(sid,)).fetchall():
                    if not self._is_explicit_fork_child_row(dict(child)):
                        frontier.append(child['id'])

    def _artifact_live_session_on_conn(self,conn,session_id):
        mapped=conn.execute('SELECT root FROM pantheon_artifact_segments WHERE session_id=?',(session_id,)).fetchone()
        if mapped is None: return None
        owner=conn.execute('SELECT erased FROM pantheon_artifact_owners WHERE root=?',(mapped['root'],)).fetchone()
        if not owner or owner['erased']: return None
        row=conn.execute('SELECT s.id FROM pantheon_artifact_segments a JOIN sessions s ON s.id=a.session_id WHERE a.root=? ORDER BY s.started_at DESC,s.id DESC LIMIT 1',(mapped['root'],)).fetchone()
        return row['id'] if row else None

    def artifact_live_session(self,session_id):
        """Resolve only a proven exact artifact alias, never a prefix/fork."""
        with self._read_ctx() as conn:
            return self._artifact_live_session_on_conn(conn,session_id)

    def _artifact_before_session_delete_on_conn(self,conn,targets):
        """One hook for ordinary single/bulk/prune/empty/delegate removal."""
        targets=set(targets)
        roots=set()
        for sid in targets:
            root=self._session_turn_lease_key_on_conn(conn,sid)
            owner=conn.execute('SELECT 1 FROM pantheon_artifact_owners WHERE root=?',(root,)).fetchone()
            if not owner: continue
            self._artifact_register_family_on_conn(conn,root,sid)
            roots.add(root)
        for root in roots:
            members={r['session_id'] for r in conn.execute('SELECT a.session_id FROM pantheon_artifact_segments a JOIN sessions s ON s.id=a.session_id WHERE a.root=?',(root,))}
            if members and members.issubset(targets):
                self._artifact_erase_roots_on_conn(conn,[root])
                continue
            # A canonical reply remains reachable only through a surviving
            # active source/clone in this exact compression owner. A branch
            # copying the same reply identity cannot keep the source visible.
            from agent.message_metadata import REPLY_SOURCE_ROW_ID_KEY, valid_reply_row_id
            remaining_sources=set()
            rows=conn.execute("SELECT m.id,m.session_id,m.display_metadata FROM messages m JOIN pantheon_artifact_segments a ON a.session_id=m.session_id WHERE a.root=? AND m.active=1 AND m.role='assistant' AND m.tool_calls IS NULL AND COALESCE(m.display_kind,'')!='hidden'",(root,)).fetchall()
            for row in rows:
                if row['session_id'] in targets: continue
                metadata=self._decode_display_metadata(row['display_metadata']) or {}
                source=metadata.get(REPLY_SOURCE_ROW_ID_KEY)
                remaining_sources.add(source if valid_reply_row_id(source) else row['id'])
            for artifact in conn.execute("SELECT artifact_id,descriptor,source_row_id FROM pantheon_artifacts WHERE root=? AND state='ready'",(root,)).fetchall():
                descriptor=json.loads(artifact['descriptor'])
                if descriptor.get('source_visible') and artifact['source_row_id'] not in remaining_sources:
                    descriptor['source_visible']=False
                    conn.execute('UPDATE pantheon_artifacts SET descriptor=? WHERE artifact_id=?',(json.dumps(descriptor),artifact['artifact_id']))
                    self._artifact_change(conn,root,artifact['artifact_id'],'visibility',descriptor)

    def artifact_begin_call(self, *, session_id, turn_id, call_id, holder):
        if not turn_id or not call_id:
            raise ArtifactError("missing_context", "Publication requires an active invocation")
        nonce = uuid.uuid4().hex
        def write(conn):
            root = self._artifact_root_on_conn(conn, session_id)
            self._check_transcript_write_guards(conn, session_id, None, turn_lease_holder=holder)
            conn.execute("INSERT OR IGNORE INTO pantheon_artifact_owners(root,cursor_key) VALUES (?,?)", (root, uuid.uuid4().hex))
            self._artifact_register_family_on_conn(conn,root,session_id)
            conn.execute("""INSERT OR IGNORE INTO pantheon_artifact_turns
                (root,turn_id,session_id,generation,holder,state,created_at)
                VALUES (?,?,?,?,?,'active',?)""", (root,turn_id,session_id,turn_id,holder,time.time()))
            turn = conn.execute("SELECT * FROM pantheon_artifact_turns WHERE root=? AND turn_id=?", (root,turn_id)).fetchone()
            if turn['state'] != 'active' or turn['holder'] != holder:
                raise ArtifactError("turn_closed", "Publication turn no longer accepts tools")
            previous = conn.execute("SELECT * FROM pantheon_artifact_calls WHERE root=? AND turn_id=? AND call_id=?", (root,turn_id,call_id)).fetchone()
            if previous:
                if previous['state'] == 'revoked':
                    raise ArtifactError('invocation_revoked', 'Invocation was cancelled')
                return root, previous['nonce']
            conn.execute("INSERT INTO pantheon_artifact_calls VALUES (?,?,?,?,'open',NULL)", (root,turn_id,call_id,nonce))
            return root, nonce
        return self._execute_write(write)

    def artifact_stage(self, *, root, session_id, turn_id, call_id, nonce, descriptor, snapshot):
        def write(conn):
            if self._artifact_root_on_conn(conn, session_id) != root:
                raise ArtifactError('source_changed','Conversation authority changed')
            call = conn.execute("SELECT * FROM pantheon_artifact_calls WHERE root=? AND turn_id=? AND call_id=?", (root,turn_id,call_id)).fetchone()
            turn = conn.execute("SELECT * FROM pantheon_artifact_turns WHERE root=? AND turn_id=?", (root,turn_id)).fetchone()
            if not call or call['nonce'] != nonce or call['state'] != 'open' or not turn or turn['state'] != 'active':
                raise ArtifactError('invocation_revoked','Publication invocation is no longer active')
            self._check_transcript_write_guards(conn, session_id, None, turn_lease_holder=turn['holder'])
            if call['artifact_id']:
                existing = conn.execute("SELECT descriptor FROM pantheon_artifacts WHERE artifact_id=?", (call['artifact_id'],)).fetchone()
                return json.loads(existing[0])
            conn.execute("""INSERT INTO pantheon_artifacts
                (artifact_id,root,turn_id,call_id,session_id,state,descriptor,snapshot,created_at)
                VALUES (?,?,?,?,?,'staged',?,?,?)""", (descriptor['artifact_id'],root,turn_id,call_id,session_id,json.dumps(descriptor),snapshot,time.time()))
            conn.execute("UPDATE pantheon_artifact_calls SET artifact_id=? WHERE root=? AND turn_id=? AND call_id=? AND nonce=?", (descriptor['artifact_id'],root,turn_id,call_id,nonce))
            return descriptor
        return self._execute_write(write)

    def artifact_finish_call(self, session_id, turn_id, call_id, accepted):
        def write(conn):
            root = self._artifact_root_on_conn(conn, session_id)
            state = 'accepted' if accepted else 'revoked'
            if not accepted:
                conn.execute("INSERT OR IGNORE INTO pantheon_artifact_calls(root,turn_id,call_id,nonce,state) VALUES (?,?,?,?,'revoked')", (root,turn_id,call_id,uuid.uuid4().hex))
            conn.execute("UPDATE pantheon_artifact_calls SET state=? WHERE root=? AND turn_id=? AND call_id=? AND state!='revoked'", (state,root,turn_id,call_id))
            if not accepted:
                conn.execute("UPDATE pantheon_artifacts SET abandoned_at=COALESCE(abandoned_at,?) WHERE root=? AND turn_id=? AND call_id=? AND state='staged'", (time.time(),root,turn_id,call_id))
        return self._execute_write(write)

    def artifact_cancel_turn(self, session_id, turn_id):
        def write(conn):
            root = self._artifact_root_on_conn(conn,session_id)
            conn.execute("INSERT OR IGNORE INTO pantheon_artifact_turns(root,turn_id,session_id,generation,state,created_at,ended_at) VALUES (?,?,?,?,'abandoned',?,?)", (root,turn_id,session_id,turn_id,time.time(),time.time()))
            conn.execute("UPDATE pantheon_artifact_turns SET state='abandoned',ended_at=? WHERE root=? AND turn_id=? AND state='active'", (time.time(),root,turn_id))
            conn.execute("UPDATE pantheon_artifact_calls SET state='revoked' WHERE root=? AND turn_id=? AND state!='revoked' AND EXISTS (SELECT 1 FROM pantheon_artifact_turns WHERE root=? AND turn_id=? AND state='abandoned')", (root,turn_id,root,turn_id))
            conn.execute("UPDATE pantheon_artifacts SET abandoned_at=COALESCE(abandoned_at,?) WHERE root=? AND turn_id=? AND state='staged'", (time.time(),root,turn_id))
        return self._execute_write(write)

    def _artifact_change(self, conn, root, artifact_id, kind, descriptor):
        conn.execute("INSERT INTO pantheon_artifact_changes(root,artifact_id,type,payload) VALUES (?,?,?,?)", (root,artifact_id,kind,json.dumps(descriptor, separators=(',',':'))))

    def _artifact_commit_on_conn(self, conn, session_id, finalization):
        """Runs after transcript INSERTs and even when the fresh batch is empty."""
        from agent.message_metadata import REPLY_SOURCE_ROW_ID_KEY, valid_reply_row_id
        root = self._artifact_root_on_conn(conn,session_id)
        turn_id = finalization['turn_id']
        turn = conn.execute("SELECT * FROM pantheon_artifact_turns WHERE root=? AND turn_id=?", (root,turn_id)).fetchone()
        if not turn:
            return []
        if turn['state'] == 'abandoned':
            raise ArtifactError('turn_cancelled','Cancelled turn cannot publish')
        if turn['generation'] != finalization['generation'] or turn['holder'] != finalization['holder']:
            raise ArtifactError('turn_changed','Turn generation or lease changed')
        selected = finalization['message']
        row_id = selected.get('_row_id')
        row = conn.execute("SELECT * FROM messages WHERE id=? AND session_id=?", (row_id,session_id)).fetchone()
        if not row or row['role'] != 'assistant' or not row['active'] or row['tool_calls'] or row['display_kind'] == 'hidden' or self._decode_content(row['content']) != selected.get('content'):
            raise ArtifactError('source_uncommitted','Exact final assistant row was not committed')
        metadata = self._decode_display_metadata(row['display_metadata']) or {}
        source = metadata.get(REPLY_SOURCE_ROW_ID_KEY)
        if not valid_reply_row_id(source):
            source = row_id
            metadata[REPLY_SOURCE_ROW_ID_KEY] = source
        captured = conn.execute("""SELECT a.* FROM pantheon_artifacts a JOIN pantheon_artifact_calls c
            ON c.root=a.root AND c.turn_id=a.turn_id AND c.call_id=a.call_id
            WHERE a.root=? AND a.turn_id=? AND a.state IN ('staged','ready') AND c.state='accepted'
            ORDER BY a.created_at,a.artifact_id""", (root,turn_id)).fetchall()
        descriptors=[]
        for artifact in captured:
            descriptor=json.loads(artifact['descriptor'])
            if artifact['state']=='ready' and descriptor.get('source_row_id') != source:
                raise ArtifactError('source_changed','A committed publication cannot bind another response')
            if artifact['state']=='staged':
                descriptor.update(source_row_id=source,source_session_id=session_id,source_visible=True)
                conn.execute("UPDATE pantheon_artifacts SET state='ready',descriptor=?,source_row_id=?,source_physical_row=? WHERE artifact_id=?", (json.dumps(descriptor),source,row_id,artifact['artifact_id']))
                self._artifact_change(conn,root,artifact['artifact_id'],'ready',descriptor)
            descriptors.append(descriptor)
        if descriptors:
            metadata['pantheon_artifacts']=descriptors
            if not str(selected.get('content') or '').strip():
                metadata['pantheon_reply_preview']='Generated '+', '.join(d['name'] for d in descriptors)+'.'
            conn.execute("UPDATE messages SET display_metadata=? WHERE id=? AND session_id=?", (self._encode_display_metadata(metadata),row_id,session_id))
        # The terminal commit is an authoritative fence for any receipt that
        # was rejected or never admitted. Retain its bytes for the grace only.
        conn.execute("UPDATE pantheon_artifact_calls SET state='revoked' WHERE root=? AND turn_id=? AND state='open'", (root,turn_id))
        conn.execute("UPDATE pantheon_artifacts SET abandoned_at=COALESCE(abandoned_at,?) WHERE root=? AND turn_id=? AND state='staged'", (time.time(),root,turn_id))
        conn.execute("UPDATE pantheon_artifact_turns SET state='committed',ended_at=? WHERE root=? AND turn_id=? AND state='active'", (time.time(),root,turn_id))
        # Pass merged metadata back only after the enclosing transaction commits.
        finalization['committed_metadata']=metadata
        return descriptors

    def artifact_changes(self, session_id, profile, *, cursor=None, since_revision=0):
        def read(conn):
            root=self._artifact_root_on_conn(conn,session_id)
            owner=conn.execute('SELECT cursor_key FROM pantheon_artifact_owners WHERE root=?',(root,)).fetchone()
            key=owner[0] if owner else ''
            if cursor:
                try:
                    encoded,signature=cursor.split('.')
                    raw=base64.urlsafe_b64decode(encoded+'='*(-len(encoded)%4))
                    if not key or not hmac.compare_digest(hmac.new(key.encode(),raw,hashlib.sha256).hexdigest(),signature):
                        raise ValueError()
                    scope=json.loads(raw)
                    if scope['root']!=root or scope['profile']!=profile:
                        raise ValueError()
                    after,through=scope['after'],scope['through']
                except (ValueError,KeyError,TypeError):
                    raise ArtifactError('invalid_cursor','Cursor does not belong to this conversation')
            else:
                if type(since_revision) is not int or since_revision<0:
                    raise ArtifactError('invalid_revision','Invalid revision')
                after=since_revision
                through=conn.execute('SELECT COALESCE(MAX(revision),0) FROM pantheon_artifact_changes WHERE root=?',(root,)).fetchone()[0]
            rows=conn.execute("SELECT * FROM pantheon_artifact_changes WHERE root=? AND revision>? AND revision<=? ORDER BY revision LIMIT 101",(root,after,through)).fetchall()
            changes=[dict(revision=r['revision'],type=r['type'],artifact=json.loads(r['payload'])) for r in rows[:100]]
            next_cursor=None
            if len(rows)>100:
                raw=json.dumps(dict(profile=profile,root=root,after=rows[99]['revision'],through=through),separators=(',',':')).encode()
                next_cursor=base64.urlsafe_b64encode(raw).decode().rstrip('=')+'.'+hmac.new(key.encode(),raw,hashlib.sha256).hexdigest()
            return dict(schema_version=1,conversation_root_id=root,changes=changes,through_revision=through,next_cursor=next_cursor)
        # Page admission and the erase fence are one serialized transaction.
        return self._execute_write(read)

    def artifact_content(self, session_id, artifact_id):
        # Eager bounded read under the database read lease. Permanent deletion
        # serializes before new admissions; already admitted responses finish.
        def read(conn):
            root=self._artifact_root_on_conn(conn,session_id)
            row=conn.execute('SELECT * FROM pantheon_artifacts WHERE artifact_id=? AND root=?',(artifact_id,root)).fetchone()
            if not row:
                raise ArtifactError('artifact_not_found','Unknown artifact',404)
            if row['state']=='deleted':
                raise ArtifactError('artifact_deleted','Artifact was removed',410)
            if row['state']!='ready':
                raise ArtifactError('artifact_not_ready','Artifact is not committed',404)
            path=self.artifact_directory / row['snapshot']
            fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
            try:
                chunks=[]; size=0
                while True:
                    chunk=os.read(fd,min(1024*1024,MAX_ARTIFACT_BYTES+1-size))
                    if not chunk: break
                    chunks.append(chunk); size+=len(chunk)
                    if size>MAX_ARTIFACT_BYTES: raise ArtifactError('snapshot_corrupt','Snapshot exceeds limit',500)
                content=b''.join(chunks)
            finally: os.close(fd)
            descriptor=json.loads(row['descriptor'])
            if len(content)!=descriptor['size'] or hashlib.sha256(content).hexdigest()!=descriptor['sha256']:
                raise ArtifactError('snapshot_corrupt','Snapshot integrity check failed',500)
            return descriptor,content

        return self._execute_write(read)

    def artifact_delete(self,session_id,artifact_id):
        def write(conn):
            root=self._artifact_root_on_conn(conn,session_id)
            row=conn.execute('SELECT * FROM pantheon_artifacts WHERE artifact_id=? AND root=?',(artifact_id,root)).fetchone()
            if not row: raise ArtifactError('artifact_not_found','Unknown artifact',404)
            if row['state']=='deleted': return row['snapshot']
            descriptor=json.loads(row['descriptor'])
            tombstone={k:descriptor[k] for k in ('artifact_id','profile','conversation_root_id')}
            if row['snapshot']:
                conn.execute('INSERT OR IGNORE INTO pantheon_artifact_cleanup VALUES (?,?)',(row['snapshot'],root))
            conn.execute("UPDATE pantheon_artifacts SET state='deleted',descriptor=?,turn_id=NULL,call_id=NULL,session_id=NULL,source_row_id=NULL,source_physical_row=NULL WHERE artifact_id=?",(json.dumps(tombstone),artifact_id))
            self._artifact_change(conn,root,artifact_id,'deleted',tombstone)
            return row['snapshot']
        snapshot=self._execute_write(write)
        self.flush_artifact_cleanup()
        return {'deleted':True,'artifact_id':artifact_id}

    def _artifact_rewind_on_conn(self,conn,session_id,removed_ids):
        if not removed_ids: return
        root=self._session_turn_lease_key_on_conn(conn,session_id)
        rows=conn.execute("SELECT * FROM pantheon_artifacts WHERE root=? AND state='ready'",(root,)).fetchall()
        removed=set(removed_ids)
        sources=set()
        for row_id in removed:
            row=conn.execute('SELECT display_metadata FROM messages WHERE id=?',(row_id,)).fetchone()
            meta=self._decode_display_metadata(row[0]) or {} if row else {}
            sources.add(meta.get('_reply_source_row_id',row_id))
        for row in rows:
            descriptor=json.loads(row['descriptor'])
            if descriptor.get('source_visible') and row['source_row_id'] in sources:
                descriptor['source_visible']=False
                conn.execute('UPDATE pantheon_artifacts SET descriptor=? WHERE artifact_id=?',(json.dumps(descriptor),row['artifact_id']))
                self._artifact_change(conn,row['root'],row['artifact_id'],'visibility',descriptor)

    def _artifact_erase_on_conn(self,conn,targets):
        roots=set(targets)
        for target in targets:
            roots.add(self._session_turn_lease_key_on_conn(conn,target))
        return self._artifact_erase_roots_on_conn(conn,roots)

    def _artifact_erase_roots_on_conn(self,conn,roots):
        snapshots=[]
        for root in roots:
            owned_snapshots=[r[0] for r in conn.execute('SELECT snapshot FROM pantheon_artifacts WHERE root=?',(root,)).fetchall() if r[0]]
            snapshots.extend(owned_snapshots)
            conn.executemany('INSERT OR IGNORE INTO pantheon_artifact_cleanup VALUES (?,?)',[(name,root) for name in owned_snapshots])
            conn.execute("INSERT INTO pantheon_artifact_owners(root,erased,cursor_key) VALUES (?,1,?) ON CONFLICT(root) DO UPDATE SET erased=1,cursor_key=excluded.cursor_key",(root,''))
            for table in ('pantheon_artifact_changes','pantheon_artifacts','pantheon_artifact_calls','pantheon_artifact_turns'):
                conn.execute('DELETE FROM '+table+' WHERE root=?',(root,))
            conn.execute('DELETE FROM session_turn_leases WHERE conversation_id=?',(root,))
        return snapshots

    def flush_artifact_cleanup(self):
        """Content-free durable unlink outbox survives post-tombstone failures."""
        with self._read_ctx() as conn:
            paths=[r[0] for r in conn.execute('SELECT snapshot FROM pantheon_artifact_cleanup LIMIT 1000').fetchall()]
        for name in paths:
            (self.artifact_directory/name).unlink(missing_ok=True)
            self._execute_write(lambda conn: conn.execute('DELETE FROM pantheon_artifact_cleanup WHERE snapshot=?',(name,)))
        return len(paths)

    def recover_artifact_captures(self, now=None):
        """Startup/housekeeping only; expiry alone never abandons a capture.

        A dead structured holder or a different current lease proves loss of
        authority. The matching generation/holder update is a SQLite CAS.
        Unreferenced old snapshots require an exclusive filesystem fence, so
        an in-flight capture between rename and INSERT remains protected.
        """
        if self.artifact_directory.is_symlink():
            raise ArtifactError('snapshot_unavailable', 'Snapshot directory must not be a symlink')
        from hermes_state import _compression_lock_holder_process_is_dead
        now=time.time() if now is None else now
        def recover(conn):
            turns=conn.execute("SELECT * FROM pantheon_artifact_turns WHERE state='active'").fetchall()
            for turn in turns:
                lease=conn.execute('SELECT holder FROM session_turn_leases WHERE conversation_id=?',(turn['root'],)).fetchone()
                dead=bool(turn['holder']) and _compression_lock_holder_process_is_dead(turn['holder'])
                superseded=lease is not None and lease['holder']!=turn['holder']
                if not dead and not superseded: continue
                changed=conn.execute("UPDATE pantheon_artifact_turns SET state='abandoned',ended_at=? WHERE root=? AND turn_id=? AND generation=? AND holder IS ? AND state='active'",(now,turn['root'],turn['turn_id'],turn['generation'],turn['holder'])).rowcount
                if not changed: continue
                if dead:
                    conn.execute('DELETE FROM session_turn_leases WHERE conversation_id=? AND holder=?',(turn['root'],turn['holder']))
                conn.execute("UPDATE pantheon_artifact_calls SET state='revoked' WHERE root=? AND turn_id=?",(turn['root'],turn['turn_id']))
                conn.execute("UPDATE pantheon_artifacts SET abandoned_at=COALESCE(abandoned_at,?) WHERE root=? AND turn_id=? AND state='staged'",(now,turn['root'],turn['turn_id']))
        self._execute_write(recover)
        removed=self.cleanup_abandoned_artifacts(now=now)
        if os.name!='posix' or not self.artifact_directory.exists(): return removed
        with capture_lock(self.artifact_directory,exclusive=True,nonblocking=True) as acquired:
            if not acquired: return removed
            def orphan_sweep(conn):
                referenced={r[0] for r in conn.execute('SELECT snapshot FROM pantheon_artifacts WHERE snapshot IS NOT NULL')}
                referenced.update(r[0] for r in conn.execute('SELECT snapshot FROM pantheon_artifact_cleanup'))
                conn.executemany('DELETE FROM pantheon_artifact_orphans WHERE snapshot=?',[(name,) for name in referenced])
                count=0
                for path in self.artifact_directory.iterdir():
                    name=path.name
                    if name in referenced or not (re.fullmatch(r'[0-9a-f-]{36}\.blob',name) or re.fullmatch(r'\.capture-[A-Za-z0-9_]+',name)): continue
                    info=path.lstat()
                    if not stat.S_ISREG(info.st_mode): continue
                    conn.execute('INSERT OR IGNORE INTO pantheon_artifact_orphans VALUES (?,?)',(name,now))
                    abandoned=conn.execute('SELECT abandoned_at FROM pantheon_artifact_orphans WHERE snapshot=?',(name,)).fetchone()[0]
                    if abandoned<now-24*3600:
                        path.unlink();count+=1
                        conn.execute('DELETE FROM pantheon_artifact_orphans WHERE snapshot=?',(name,))
                return count
            removed+=self._execute_write(orphan_sweep)
        return removed

    def cleanup_abandoned_artifacts(self, now=None):
        """Never called by list/reconnect. Only explicit abandonment qualifies."""
        cutoff=(time.time() if now is None else now)-24*3600
        def write(conn):
            rows=conn.execute("SELECT artifact_id,snapshot FROM pantheon_artifacts WHERE state='staged' AND abandoned_at IS NOT NULL AND abandoned_at<?",(cutoff,)).fetchall()
            for row in rows:
                conn.execute('INSERT OR IGNORE INTO pantheon_artifact_cleanup(snapshot,root) SELECT snapshot,root FROM pantheon_artifacts WHERE artifact_id=?',(row['artifact_id'],))
                conn.execute("UPDATE pantheon_artifacts SET state='abandoned',snapshot=NULL,descriptor=NULL WHERE artifact_id=?",(row['artifact_id'],))
            return [row['snapshot'] for row in rows if row['snapshot']]
        paths=self._execute_write(write)
        self.flush_artifact_cleanup()
        return len(paths)
