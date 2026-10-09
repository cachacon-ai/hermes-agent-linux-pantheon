# Persisted assistant reply identity

Pantheon PAN-13 uses exact reply identity for Mark unread and archived-chat restoration. A live completion and the subsequent history snapshot must identify the same authored assistant reply, including identical answer text and replies produced while Pantheon was disconnected.

Every `message.complete` payload advertises `reply_identity_version: 1`. A successfully persisted, visible final assistant reply adds:

- `row_id`: its current physical SQLite message row, matching `session.history.messages[].row_id`.
- `source_row_id`: the original committed assistant row identity, matching `session.history.messages[].source_row_id` and preserved through in-place compaction and compression continuation clones.
- `timestamp`: the persisted creation time in Unix seconds, matching history.
- `reply_text`: the persisted reply's existing history display projection. This can differ from `text`, which still includes delivery-only output hooks and verifier footers.

Consumers qualify identity by profile and durable conversation, then compare `source_row_id`. Do not deduplicate by text, timestamp, or the physical `row_id`: repeated replies can share text and timestamps, and compaction changes physical row IDs.

The source is stored in the existing `display_metadata._reply_source_row_id` sidecar. It is excluded from provider messages and rough context estimates; it does not change prompts or caching. The original row ID is stamped within the same SQLite transaction as the assistant insertion. Append-only agent dictionaries receive it through the existing post-commit marker synchronization. Before advertising identity, the finalizer also reads the exact selected active SQLite row and validates its source and display fields; a failed compaction transaction's stale in-memory markers cannot authorize a nonexistent or different row.

Failed, interrupted, unsaved, empty and synthetic completions still advertise version 1 but omit row identity. Presence of the version makes absence authoritative; consumers must not manufacture an assistant reply from error text. History remains the authoritative recovery surface.

Existing legacy rows acquire their actual current row identity as their source when decoded, and that identity survives future compactions. Clones produced before this upgrade have no recorded original provenance; the upgrade cannot reconstruct it without guessing. An older server without this contract is unsupported for exact unread/archive reconciliation; a Hermes server must provide this capability before the dependent Pantheon behavior is enabled. Live Hermes already runs this code (shipped with PAN-26; its state DB is on schema 29), so merging it aligns `main` with production.

Validation uses the real SessionDB batch writer, actual SQLite compaction/rotation transactions, finalizer, history projection and `_run_prompt_submit` event path with a deterministic agent, without model/network calls. Focused tests cover repeated identical replies at identical timestamps, hooks/footers, rollback, legacy rows, concurrent compression tails, blank-tail repair collisions and failure omissions.
