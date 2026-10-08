# Pantheon generated files (PAN-26)

## Capability and publication

Pantheon requests `source: "pantheon", artifact_delivery_version: 1` when creating a session. Hermes returns `artifact_delivery_version`, `conversation_root_id` and the resolved `profile`. Resume restores the original capability; existing legacy sessions never acquire tools or a changed cached prompt. Version zero includes `artifact_delivery_unavailable_code` and `artifact_delivery_unavailable_reason`. Only `legacy_session` can be addressed by creating a new conversation. `unsupported_protocol` requires a compatible installed Codex adapter, `unsupported_source` requires a supported producing environment, and `invalid_configuration` requires correcting that profile's configuration.

Only the session-scoped `pantheon_publish_artifact(path, name?)` tool publishes files. Prose, MEDIA tags, filenames and tool-complete callbacks confer no authority. The current turn receives a private output directory before tools run, outside the cached system prefix. Explicit selected workspaces and `pantheon.artifact_source_roots` may add trusted absolute roots. Generic publication requires the actual existing local terminal environment. Remote backends have no safe-reader adapter and remain explicitly unavailable. Native Codex uses its own host/cwd and a parent dynamic-tool callback validated against the exact active thread, turn and call. A schema-only probe of the actual CLI gates that protocol without starting a provider turn.

Capture rejects nonregular files, symlinks in every path component, protected configuration paths, directory traversal, source mutations and bytes beyond 25 MiB. It supports self-contained HTML, PNG/JPEG/GIF/WebP, PDF, UTF-8 text/Markdown/CSV/JSON/RTF and validated DOCX/XLSX/PPTX packages. Original workspace files are never modified or removed.

## Durability and identity

The coordinator freezes profile/session/turn/environment/root authority before scheduling a worker. Immutable bytes are fsynced and renamed into profile-local private snapshots, then recorded as staged in that profile's SessionDB. An executor receipt must accept the call; timeout or cancellation revokes it even if a worker starts or finishes late. Repeating the same invocation returns its original snapshot; a distinct call creates another artifact.

The final source row and ready transition commit in the same transcript writer transaction, including a zero-fresh-row recovery flush. Tentative canonical metadata is copied before stamping, so a rolled-back transaction cannot leave a source ID in the live message that a retry reuses after another writer. Canonical PAN-13 `source_row_id` survives compression; branches retain reply provenance but strip publication metadata and own no parent artifacts. Rewind retains hub bytes while changing source visibility. File-only final answers retain empty provider content and obtain a display-only preview. Completion/history expose `artifact_only`, `provider_text`, `presentation_text`, stable delivery identity and descriptors. Persistence fields are excluded from main, auxiliary and transport provider projections.

Stop first publishes the runtime interrupt and current-turn cancellation fence, then signals the native runtime, socket, tool workers and child agents. Durable artifact cancellation is best effort after those signals. A missing conversation or unavailable database cannot prevent Stop, and clearing an ordinary interrupt cannot revive publication for that cancelled turn.

Schema version 29 records content-free segment-to-root ownership independently of removable session ancestry. Ordinary deletion, bulk deletion, pruning and empty-session maintenance preserve the original artifact root while any compression segment survives. The removed exact root remains a read/resume alias to the surviving conversation; repeating ordinary DELETE remains idempotent and cannot delete that survivor or an unrelated prefix match. Explicit branches retain separate ownership.

Deleting a segment hides a generated chat card only when its canonical assistant source no longer has an active reachable clone in the surviving compression conversation. The visibility change is durable and replayable; a branch clone cannot keep the parent card visible. Removing the last owning segment erases registry/feed/call/turn payloads, cursor authority and lease state, and queues its bytes for durable unlinking. Later permanent deletion accepts an already removed root alias and erases all surviving segments of that owner. Empty-session maintenance preserves staged captures as resumable content.

## Scoped API

All routes use the existing authenticated profile/session middleware:

- `GET /api/sessions/{session}/artifacts?profile=...&since_revision=...` returns schema 1, the canonical root, immutable changes, a pinned `through_revision` and an optional opaque `next_cursor`. Follow pages with the cursor unchanged. Pages contain at most 100 changes.
- `GET /api/sessions/{session}/artifacts/{artifact}/content?profile=...` returns verified immutable bytes with attachment, no-store and nosniff headers. HTML also has a restrictive sandbox CSP.
- `DELETE /api/sessions/{session}/artifacts/{artifact}?profile=...` is idempotent. Deleted owned content returns 410; foreign artifacts return 404.

Page/content admission serializes with the permanent deletion fence. Bytes already admitted to a response may finish; subsequent admissions fail. Permanent conversation deletion purges registry and changefeed descriptions and queues retryable byte removal, leaving content-free fences. Snapshot deletion failures remain in a durable unlink outbox.

## Recovery and retention

Backend startup and hourly housekeeping enumerate the existing local default and named-profile catalog, independently of heartbeat or orphan-sweep opt-outs. Each dedicated existing-only SessionDB handle closes after recovery; caller-owned launch/resume handles remain open. Missing databases are never bootstrapped, deleted profiles and symlinked paths are skipped, and one profile's failure cannot stop the others. Generation/holder CAS recovers only proven-dead or superseded lease holders. Lease expiry alone is insufficient. Listing and reconnect never discard staging. Call revocation and terminal commit mark rejected captures abandoned. Abandoned snapshots get a 24-hour grace before collection; ready files remain until artifact deletion or deletion of the last owning conversation segment.

A shared filesystem capture lock covers snapshot creation through the staging INSERT. Orphan recovery takes an exclusive nonblocking lock, records content-free discovery timestamps and waits a new 24-hour grace before unlinking unreferenced blobs or capture temporaries. This protects the rename-to-SQLite crash boundary and in-flight captures. Recovery operates on explicit profile databases within the backend's local catalog; profiles on another machine require that machine's backend housekeeping.

## Verification boundary

Use `scripts/run_tests.sh` with a disposable HERMES_HOME. Focused regression coverage includes real executor acceptance and timeout paths, full conversation-loop attachment-only termination, atomic rollback/retry, immutable snapshots, scoped pagination, compression, rewind, deletion, durable cleanup failure/retry, live/dead/superseded lease recovery, orphan capture fencing, actionable capabilities and native dynamic-tool identity checks. Local CLI schema support does not establish deployed PC support or a live native-provider publication turn. Deployments and physical iPhone acceptance are separate from local source/runtime verification.
