# Fork `main` Python test failures (post #19)

**Context:** CI run [38297639134](https://github.com/cachacon-ai/hermes-agent-linux-pantheon/actions/runs/38297639134) on `main` at `a0521ded` (merge of #19). Eight failures in seven files; reproduced on current `main` with targeted `scripts/run_tests.sh` / `pytest` (not a full-suite run).

**Pantheon pin today:** `HERMES_RUNTIME_PIN=a8c7d106` (one commit behind `main`; only #19 CI runner docs differ).

---

## Failure summary

| # | Test | CI error (short) | Local repro | Root cause | Introduced by | Class | #17 | #18 | Pantheon @ pin |
|---|------|------------------|-------------|------------|---------------|-------|-----|-----|----------------|
| 1 | `tests/agent/test_compression_concurrent_fork.py::test_rotation_publish_failure_restores_proactive_prune_runway` | `model_config` has extra `max_iterations: 9223372036854775807` | Yes (via `run_tests.sh`; stable) | Rotation abort restores messages + in-memory prune runway but not parent `model_config`. A compression side path calls `create_session` / merge with `agent._session_init_model_config` (includes `max_iterations=sys.maxsize`), polluting the row. | Test: `bf6a210ab` (proactive prune). Pollution: lazy session enrichment (`run_agent._ensure_db_session` + `hermes_state` merge). | **Real bug** (rollback incomplete) | No | **Partial** — relaxes assertion to check only `keep` + `_proactive_prune_rearm_tokens` | **Yes** — compression + `SessionDB` vendored; long sessions hit compression |
| 2 | `tests/gateway/test_resume_command.py::TestHandleSessionsCommand::test_sessions_full_lists_conversations_created_by_gateway_resets` | Second `create_session` replaces `max_iterations` 60→999 | Yes | `hermes_state._insert_session_row` `ON CONFLICT` only preserves `_reset_from` when existing `model_config` is *only* `{_reset_from}`. After enrichment, later upserts take the `ELSE excluded.model_config` branch (full replace). | Test contract: `ce89afa59` / `5a10537b2` (reset lineage). SQL shape predates test. | **Real bug** | No | No | **Yes** — gateway + `SessionDB` vendored; `/sessions` / reset flows |
| 3 | `tests/hermes_cli/test_config_read_guard.py::test_no_raw_config_yaml_reads_outside_owner_modules` | Offender: `agent/pantheon_artifacts.py:448` `yaml.safe_load` near `config.yaml` | Yes | `capability_details()` reads producing profile config with raw `yaml.safe_load` instead of `load_config_readonly()` / `read_user_config_raw()`. | `b2459b634` (PAN-26 pantheon artifacts) | **Fork divergence** (policy + Pantheon surface) | No | No | **Yes** — `agent/pantheon_artifacts.py`, `tools/pantheon_tool.py` at pin |
| 4 | `tests/hermes_cli/test_local_quickstart.py::test_quickstart_runs_all_three_legs` | `POST …/quickstart` → **409** | Yes | Preflight: `select_variant()` returns no fit on CI/this VM → `409` before stubs run. Tests don't mock catalog/hardware preflight. | Quickstart route/tests (dashboard local models); exposed when runners can't serve default catalog entry | **Stale test** (missing preflight stubs) | **Yes** — `autouse` `_quickstart_catalog_preflight` stubs `select_variant` + `_engine_too_old` | No | **Low** — dashboard local-models quickstart; not core Pantheon chat path unless UI uses it |
| 5 | `tests/hermes_cli/test_local_quickstart.py::test_quickstart_skips_satisfied_legs` | Same **409** | Yes | Same as #4 | Same | **Stale test** | **Yes** (same fixture) | No | **Low** (same) |
| 6 | `tests/test_tui_gateway_server.py::test_session_resume_passes_stored_runtime_to_agent` | `KeyError: 'model_override'` on `server._sessions[…]` | Yes | Test's `fake_init_session` omits `model_override`; production `_init_session` stores it (`tui_gateway/server.py` ~9798). Overrides reach `_make_agent` but session dict isn't mirrored in the mock. | Test expectation tightened vs `_stored_session_runtime_overrides`; not a runtime regression in `_init_session` | **Stale test** | **Yes** — adds `model_override` to fake `_init_session` | No | **Yes** — Pantheon uses `hermes serve` / `tui_gateway` JSON-RPC; session resume + overrides |
| 7 | `tests/tools/test_subprocess_stdin_guard.py::test_all_tui_subprocess_calls_have_stdin` | Missing `stdin=` on `codex_app_server_session.py:86` | Yes | `supports_pantheon_dynamic_tools()` probes Codex via `subprocess.run` without `stdin=subprocess.DEVNULL`. | `b2459b634` (Pantheon Codex protocol probe) | **Fork divergence** (stdin guard policy) | No | **Yes** — adds `stdin=subprocess.DEVNULL` | **Yes** — probe in vendored `agent/transports/codex_app_server_session.py` at pin |
| 8 | `tests/acp/test_session.py::TestPersistence::test_assistant_reasoning_fields_persisted` | Extra `display_metadata._reply_source_row_id` on restore | Yes | `hermes_state._rows_to_conversation` stamps reply-source row id on assistant loads (`REPLY_SOURCE_ROW_ID_KEY`); test compares exact dict without popping `display_metadata`. Reasoning fields themselves persist. | Reply identity load path (`b2459b634` fork PAN-26 batch; upstream reply-identity pattern) | **Stale test** | No | No | **No** — ACP adapter not Pantheon's primary surface; code at pin if ACP used |

---

## Fix plan (Hermes fork first)

### Phase A — Land PAN-44 batches (unblock CI quickly)

1. **Merge #17 (`pan44-batch-a`)** after rebase on `main`
   - Fixes: quickstart (#4–5), TUI `model_override` test (#6).
   - Also carries FAL/video test hardening (unrelated to this failure set).

2. **Merge #18 (`pan44-batch-b`)** on top of #17
   - Fixes: subprocess stdin guard (#7).
   - **Compression (#1):** only assertion relaxation — treat as stopgap; schedule Phase B.1.

3. **Merge #20** when ready (orthogonal to pytest failures; publish/deploy gating).

### Phase B — Code fixes still needed after batches

| Item | Files | Approach | Proof test |
|------|-------|----------|------------|
| **B.1 Compression `model_config` rollback** | `agent/conversation_compression.py`, possibly `run_agent.py` / `hermes_state.py` | Snapshot parent `model_config` at rotation attempt start; on publish failure restore it alongside messages + prune runway. Avoid relying on weakened assertion. | `test_rotation_publish_failure_restores_proactive_prune_runway` (strict equality restored) |
| **B.2 Gateway reset `model_config` merge** | `hermes_state.py` (`_insert_session_row` `ON CONFLICT`) | When row has `_reset_from`, merge incoming keys without clobbering established fields on bare retry upserts (match test comment: enrich once, don't replace). | `test_sessions_full_lists_conversations_created_by_gateway_resets` |
| **B.3 Pantheon config read** | `agent/pantheon_artifacts.py` | Replace raw `yaml.safe_load(path…)` with `load_config_readonly()` scoped to `profile_home` (or `read_user_config_raw` if truly raw diagnostic). | `test_no_raw_config_yaml_reads_outside_owner_modules` |
| **B.4 ACP persistence test** | `tests/acp/test_session.py` | Pop `display_metadata` (or assert `_reply_source_row_id` present) like other restore tests. | `test_assistant_reasoning_fields_persisted` |

### Phase C — Pantheon runtime pin

1. After fork `main` is green (Phases A + B), bump `HERMES_RUNTIME_PIN` in Pantheon (`scripts/sync-hermes-runtime.sh`) to the fixing commit.
2. Run Pantheon-focused subset (below) + Pantheon's own vitest/e2e as applicable.
3. **No pin bump required** for failures that are test-only (#4–6, #8) once CI passes on fork; **pin bump required** for #1–3, #7 behavior in vendored code.

---

## Scoping fork Python CI (Pantheon-relevant, fast PRs)

**Goal:** PR Python lane finishes in minutes on `ubuntu-latest` + 4 workers; full Hermes suite on a schedule or manual workflow.

### Proposed repository variables

| Variable | Purpose | Example fork value |
|----------|---------|-------------------|
| `CI_PYTHON_MODE` | `full` (default) \| `pantheon` \| `paths` | `pantheon` |
| `CI_PYTHON_TEST_PATHS` | When `paths`: newline or comma list of paths passed to `run_tests.sh` | `tests/tui_gateway/\ntests/agent/test_pantheon_*\n…` |
| `CI_PYTHON_EXTRA_ARGS` | Optional pytest `-k` / markers | `-m pantheon` (if markers added) |
| `CI_PYTHON_FULL_CRON` | Cron for scheduled full suite | `0 6 * * *` |
| Existing `CI_TEST_WORKERS`, `CI_TEST_TIMEOUT_MINUTES`, `CI_LINUX_RUNNER` | Keep from #19 / `docs/ci-fork.md` | `4`, `120`, `ubuntu-latest` |

### Suggested `pantheon` path set (initial)

**Product code touched by Pantheon (indicative):**

- `tui_gateway/`, `agent/pantheon_artifacts.py`, `agent/transports/codex_app_server_session.py`
- `tools/pantheon_tool.py`, `hermes_state.py` (session/compression edges), `gateway/` (subset), `hermes_cli/web_server.py` + `hermes_cli/web_routers/`
- `agent/conversation_compression.py`, `run_agent.py` (session resume / compression)

**Tests:**

- `tests/tui_gateway/test_pantheon_*`, `tests/agent/test_pantheon_*`, `tests/agent/transports/test_pantheon_*`
- `tests/test_tui_gateway_server.py` (large file — consider splitting or `-k pantheon` marker later)
- `tests/gateway/test_resume_command.py` (reset listing contract)
- `tests/hermes_cli/test_config_read_guard.py`, `tests/tools/test_subprocess_stdin_guard.py`
- `tests/agent/test_compression_concurrent_fork.py` (compression rollback class)

**Workflow shape:**

```yaml
# Pseudocode in tests.yml Run tests step
if vars.CI_PYTHON_MODE == 'pantheon':
  scripts/run_tests.sh tests/tui_gateway/ tests/agent/test_pantheon_artifacts.py …
elif vars.CI_PYTHON_MODE == 'paths':
  scripts/run_tests.sh ${{ vars.CI_PYTHON_TEST_PATHS }}
else:
  scripts/run_tests.sh
```

Add `workflow_dispatch` + weekly `schedule` job: `CI_PYTHON_MODE=full` on same runner vars.

Extend `scripts/ci/classify_changes.py` with a `pantheon` lane so PRs touching only `website/` don't run either Python mode incorrectly.

### What you lose with `pantheon` mode on PRs

- Coverage of ~38k tests: most messaging adapters, desktop, plugins, skills, cron, kanban, provider matrix, etc.
- Regressions in shared core only exercised by distant tests (e.g. obscure `hermes_cli` commands).
- Confidence that fork == upstream for non-Pantheon surfaces.

### Mitigations

- Scheduled nightly/weekly **full** run on fork `main`.
- Require full green before **pin bump** to Pantheon (release gate).
- Optionally run full suite on PRs that touch outside pantheon path set (classifier fail-open for unknown paths — already in `classify_changes.py` policy).

---

## Verification commands (targeted)

```bash
source .venv/bin/activate
scripts/run_tests.sh tests/acp/test_session.py -k test_assistant_reasoning_fields_persisted
scripts/run_tests.sh tests/agent/test_compression_concurrent_fork.py -k test_rotation_publish_failure_restores_proactive_prune_runway
scripts/run_tests.sh tests/gateway/test_resume_command.py -k test_sessions_full_lists_conversations_created_by_gateway_resets
scripts/run_tests.sh tests/hermes_cli/test_config_read_guard.py
scripts/run_tests.sh tests/hermes_cli/test_local_quickstart.py
scripts/run_tests.sh tests/test_tui_gateway_server.py -k test_session_resume_passes_stored_runtime_to_agent
scripts/run_tests.sh tests/tools/test_subprocess_stdin_guard.py
```

---

## References

- Open PRs: [#17](https://github.com/cachacon-ai/hermes-agent-linux-pantheon/pull/17) batch A, [#18](https://github.com/cachacon-ai/hermes-agent-linux-pantheon/pull/18) batch B, [#20](https://github.com/cachacon-ai/hermes-agent-linux-pantheon/pull/20) `CI_PRIMARY_REPOSITORY`
- Fork CI variables: `docs/ci-fork.md` (#19)
