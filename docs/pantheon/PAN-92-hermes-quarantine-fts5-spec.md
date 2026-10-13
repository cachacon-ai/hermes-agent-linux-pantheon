# PAN-92 / Hermes pin `b4dd3e00` — FTS5 projection test & quarantine spec

**Scope:** Investigation + fork fix A.1 (behavioral FTS5 projection test). Fork `cachacon-ai/hermes-agent-linux-pantheon`.  
**Pantheon:** `cachacon-ai/pantheon` @ `pantheon-v1` / `1c500511`.  
**Quarantine source:** `tests/pantheon/hermes_quarantine.yaml` (43 entries, `PANTHEON_CI=1` via `scripts/run_tests_parallel.py`).

---

## Executive summary

| Metric | Value |
|--------|--------|
| Quarantine entries | **43** |
| `pantheon-layout` (keep) | **42** |
| `fork-fixable` (fix in fork, then remove from yaml) | **1** |
| `stale-removable` without fork work | **0** |
| Phase B “8 fixed on fork” nodeids **in** this yaml | **0** — Pantheon already pins `b4dd3e00`; those tests are **not** quarantined and are expected to pass in the vendored full suite |

**PR #92 +7 (36 → 43):** The only entry that targets Hermes **runtime/session** behavior (not image/docs layout) is #43 (FTS5 projection trace test). The other six additions vs a 36-entry baseline are almost certainly more **website / oauth / docs** contract tests in the same family as rows 25–42 (exact +7 diff requires Pantheon git history).

---

## Part A — Fork fixes (Hermes fork first)

### A.1 FTS5 projection test — **fork-fixable** (quarantine #43)

| Question | Answer |
|----------|--------|
| **Nodeid** | `tests/test_hermes_state.py::TestFTS5Search::test_search_projection_skips_context_enrichment_queries` |
| **Pantheon yaml reason** | Claims pin `b4dd3e00` “no longer emits” `WITH TARGET AS` SQL. |
| **Correction** | SQL **still** runs in `hermes_state_search.py::_finalize_search_matches()` (`WITH target AS (`). Trace misses it because enrichment uses `SessionDB._read_ctx()` (pooled WAL read conns), not the connections the test traces (`94869e5a0` / `0534f1033`). |
| **Search regression for Pantheon?** | **No** — behavior (skip context when `fields` omits `"context"`) is intact. |
| **Planned fix** | **Files:** `tests/test_hermes_state.py` only. Replace SQL-trace assertions with behavioral checks (no `context` key on projected rows; non-empty `context` when field requested or default). Do **not** revert read pool. |
| **Quarantine** | **Remove** from yaml after fork test fix is in the vendored pin (still `b4dd3e00` or next bump). |

### A.2 Phase B “8 fixed on fork” vs this quarantine list

Investigation doc failures (compression rollback, gateway reset merge, pantheon config read guard, quickstart×2, TUI `model_override`, stdin guard, ACP `display_metadata`) were fixed on fork **`b4dd3e00`**.

**Verification against attached `hermes_quarantine.yaml`:** **None of those eight nodeids appear.** Pantheon does not quarantine them at pin `b4dd3e00`; they should run in the default vendored suite. No yaml trim is needed for A.2—only confirm green in Pantheon CI.

| Nodeid (Phase B) | In quarantine yaml? |
|------------------|---------------------|
| `tests/agent/test_compression_concurrent_fork.py::test_rotation_publish_failure_restores_proactive_prune_runway` | **No** |
| `tests/gateway/test_resume_command.py::TestHandleSessionsCommand::test_sessions_full_lists_conversations_created_by_gateway_resets` | **No** |
| `tests/hermes_cli/test_config_read_guard.py::test_no_raw_config_yaml_reads_outside_owner_modules` | **No** |
| `tests/hermes_cli/test_local_quickstart.py::test_quickstart_runs_all_three_legs` | **No** |
| `tests/hermes_cli/test_local_quickstart.py::test_quickstart_skips_satisfied_legs` | **No** |
| `tests/test_tui_gateway_server.py::test_session_resume_passes_stored_runtime_to_agent` | **No** |
| `tests/tools/test_subprocess_stdin_guard.py::test_all_tui_subprocess_calls_have_stdin` | **No** |
| `tests/acp/test_session.py::TestPersistence::test_assistant_reasoning_fields_persisted` | **No** |

---

## Part B — Pantheon quarantine trims (after fork fix)

### B.1 Trims at pin `b4dd3e00` (already pinned)

- **Do not** expect to remove 8 lines for Phase B—those tests were never on this list.
- **Do** remove **1** line after A.1 lands: row **#43** (FTS5 projection).
- **Keep** rows **#1–#42** unless Pantheon changes deployment to vendor Hermes root `Dockerfile`, `docker/tini-shim.sh`, `docker/stage2-hook.sh`, and Hermes `website/` tree (unlikely per `HERMES-RUNTIME.md`).

### B.2 When an entry could move from `pantheon-layout` to removable

Only if Pantheon CI starts shipping the asserted artifacts (e.g. copy `website/static/oauth/client-metadata.json` from Hermes, or align `docker/stage2-hook.sh` with Hermes). That is a **product/layout** decision, not a fork code fix.

---

## Part C — Full 43-entry classification

**Legend**

| Tag | Meaning |
|-----|---------|
| `pantheon-layout` | Legitimate skip: test asserts Hermes monorepo paths Pantheon does not ship; keep quarantined under `PANTHEON_CI=1`. |
| `fork-fixable` | Fix in `hermes-agent-linux-pantheon`; remove from yaml once fix is vendored. |
| `stale-removable` | Remove without layout change (dead nodeid, or quarantine masking already-fixed behavior). |

| # | Pytest nodeid | Tag | Notes / planned fix |
|---|---------------|-----|---------------------|
| 1 | `tests/skills/test_office_document_skills.py::test_docs_pages_generated` | `pantheon-layout` | Asserts `website/docs` pages; Pantheon website is Pantheon-owned. Keep. |
| 2 | `tests/test_tini_shim.py::test_does_not_double_wrap_existing_wrapper` | `pantheon-layout` | Needs `docker/tini-shim.sh` from Hermes image layout. Keep. |
| 3 | `tests/test_tini_shim.py::test_shim_script_is_executable_bit_friendly` | `pantheon-layout` | Same. Keep. |
| 4 | `tests/test_tini_shim.py::test_strips_g_and_double_dash` | `pantheon-layout` | Same. Keep. |
| 5 | `tests/test_tini_shim.py::test_strips_p_and_e_with_arguments` | `pantheon-layout` | Same. Keep. |
| 6 | `tests/tools/test_dockerfile_immutable_install.py::test_dockerfile_bakes_code_scoped_install_method_stamp` | `pantheon-layout` | Hermes root `Dockerfile` install stamps; Pantheon service-only image. Keep. |
| 7 | `tests/tools/test_dockerfile_immutable_install.py::test_dockerfile_bakes_photon_sidecar_deps` | `pantheon-layout` | Hermes root Dockerfile photon sidecar. Keep. |
| 8 | `tests/tools/test_dockerfile_immutable_install.py::test_dockerfile_makes_opt_hermes_readonly_for_hermes_user` | `pantheon-layout` | Hermes root Dockerfile permissions. Keep. |
| 9 | `tests/tools/test_dockerfile_immutable_install.py::test_dockerfile_redirects_lazy_installs_to_durable_target` | `pantheon-layout` | Hermes root Dockerfile lazy-install redirect. Keep. |
| 10 | `tests/tools/test_dockerfile_node_modules_perms.py::test_dockerfile_uses_prebuilt_tui_instead_of_writable_node_modules` | `pantheon-layout` | Asserts `HERMES_TUI_DIR` in Hermes root Dockerfile. Keep. |
| 11 | `tests/tools/test_dockerfile_pid1_reaping.py::test_dispatcher_non_pid1_fallback_restores_s6_helpers_on_path` | `pantheon-layout` | Hermes docker entrypoint/init wiring. Keep. |
| 12 | `tests/tools/test_dockerfile_pid1_reaping.py::test_dockerfile_entrypoint_routes_through_the_init` | `pantheon-layout` | Same. Keep. |
| 13 | `tests/tools/test_dockerfile_pid1_reaping.py::test_dockerfile_installs_an_init_for_zombie_reaping` | `pantheon-layout` | Same. Keep. |
| 14 | `tests/tools/test_stage2_hook_api_server_keygen.py::test_keygen_appends_to_existing_env_without_key` | `pantheon-layout` | Hermes `docker/stage2-hook.sh` api_server keygen block. Keep. |
| 15 | `tests/tools/test_stage2_hook_api_server_keygen.py::test_keygen_creates_env_when_missing` | `pantheon-layout` | Same. Keep. |
| 16 | `tests/tools/test_stage2_hook_api_server_keygen.py::test_keygen_env_key_drops_stale_empty_assignment` | `pantheon-layout` | Same. Keep. |
| 17 | `tests/tools/test_stage2_hook_api_server_keygen.py::test_keygen_env_key_with_existing_env_file_key_warns_not_clobbers` | `pantheon-layout` | Same. Keep. |
| 18 | `tests/tools/test_stage2_hook_api_server_keygen.py::test_keygen_never_overwrites_operator_key` | `pantheon-layout` | Same. Keep. |
| 19 | `tests/tools/test_stage2_hook_api_server_keygen.py::test_keygen_readonly_env_degrades_to_warning_not_boot_abort` | `pantheon-layout` | Same. Keep. |
| 20 | `tests/tools/test_stage2_hook_api_server_keygen.py::test_keygen_refuses_symlinked_env` | `pantheon-layout` | Same. Keep. |
| 21 | `tests/tools/test_stage2_hook_api_server_keygen.py::test_keygen_skips_when_container_env_provides_key` | `pantheon-layout` | Same. Keep. |
| 22 | `tests/tools/test_stage2_hook_api_server_keygen.py::test_keygen_warns_on_weak_container_env_key` | `pantheon-layout` | Same. Keep. |
| 23 | `tests/tools/test_stage2_hook_api_server_keygen.py::test_keygen_weak_env_key_warning_suppressed_when_env_file_key_wins` | `pantheon-layout` | Same. Keep. |
| 24 | `tests/tools/test_stage2_hook_symlink_chown.py::test_stage2_skips_recursive_repairs_when_tree_is_already_owned` | `pantheon-layout` | Hermes stage2 `tree_has_non_hermes_owner()`; Pantheon hook differs. Keep. |
| 25 | `tests/website/test_extract_skills.py::test_source_url_synthesizes_clawhub` | `pantheon-layout` | Pantheon `website/scripts/extract_skills.py` ≠ Hermes docs contract. Keep. |
| 26 | `tests/website/test_extract_skills.py::test_source_url_synthesizes_clawhub_strips_prefix` | `pantheon-layout` | Same. Keep. |
| 27 | `tests/website/test_extract_skills.py::test_source_url_synthesizes_clawhub_with_owner` | `pantheon-layout` | Same. Keep. |
| 28 | `tests/website/test_extract_skills.py::test_source_url_synthesizes_clawhub_with_owner_strips_prefix` | `pantheon-layout` | Same. Keep. |
| 29 | `tests/website/test_generate_llms_txt.py::test_bot_mode_is_reachable` | `pantheon-layout` | Hermes Docusaurus `llms.txt` generator; Pantheon website not vendored from fork. Keep. |
| 30 | `tests/website/test_generate_llms_txt.py::test_curation_orders_pages_without_gatekeeping_them` | `pantheon-layout` | Same. Keep. |
| 31 | `tests/website/test_generate_llms_txt.py::test_every_docs_page_is_indexed` | `pantheon-layout` | Same. Keep. |
| 32 | `tests/website/test_generate_llms_txt.py::test_every_indexed_page_exists` | `pantheon-layout` | Same. Keep. |
| 33 | `tests/website/test_generate_llms_txt.py::test_mdx_pages_are_indexed_without_their_imports` | `pantheon-layout` | Same. Keep. |
| 34 | `tests/website/test_generate_llms_txt.py::test_per_skill_catalog_pages_stay_out` | `pantheon-layout` | Same. Keep. |
| 35 | `tests/website/test_generate_llms_txt.py::test_section_landing_pages_resolve_to_their_directory` | `pantheon-layout` | Same. Keep. |
| 36 | `tests/website/test_generate_llms_txt.py::test_the_enumerator_sees_the_whole_docs_tree` | `pantheon-layout` | Same. Keep. |
| 37 | `tests/skills/test_xurl_x_search_routing.py::test_x_search_doc_separates_discovery_from_account_actions` | `pantheon-layout` | Asserts `website/docs/.../x-search.md` prose. Keep. |
| 38 | `tests/tools/test_mcp_cimd.py::test_document_client_id_is_the_url_hermes_sends` | `pantheon-layout` | Needs `website/static/oauth/client-metadata.json` (Hermes docs site). Keep. |
| 39 | `tests/tools/test_mcp_cimd.py::test_document_declares_every_callback_hermes_can_build` | `pantheon-layout` | Same. Keep. |
| 40 | `tests/tools/test_mcp_cimd.py::test_document_advertises_a_public_native_client` | `pantheon-layout` | Same. Keep. |
| 41 | `tests/tools/test_mcp_cimd.py::test_document_carries_no_shared_secret` | `pantheon-layout` | Same. Keep. |
| 42 | `tests/tools/test_mcp_cimd.py::test_generated_redirect_uri_is_registered_in_the_document` | `pantheon-layout` | Same. Keep. |
| 43 | `tests/test_hermes_state.py::TestFTS5Search::test_search_projection_skips_context_enrichment_queries` | `fork-fixable` | Stale trace-based test (read pool); not layout. **Fix:** `tests/test_hermes_state.py` behavioral assertions. **Then** remove from yaml. Correct Pantheon `reason` field when editing yaml. |

### Part C rollup

| Tag | Count | Action |
|-----|-------|--------|
| `pantheon-layout` | 42 | Retain in `hermes_quarantine.yaml` for `PANTHEON_CI=1`; review quarterly per yaml header. |
| `fork-fixable` | 1 | Row 43 only. |
| `stale-removable` | 0 | Nothing to drop solely because pin is already `b4dd3e00`. |

---

## Appendix — FTS5 commits (unchanged from prior analysis)

| Topic | Commit |
|-------|--------|
| CTE context neighbors | `aebf32229` |
| `fields=` projection gating | `ffb54305c` |
| Trace-based test | `f795d542f` |
| Search mixin split | `21c7ae856` |
| Read pool / `_read_ctx` | `94869e5a0`, `0534f1033` |

---

## Summary

| Item | Verdict |
|------|---------|
| FTS5 quarantine #43 | **fork-fixable** test fix; **no** Pantheon search regression |
| Phase B eight nodeids | **Not quarantined** at pin `b4dd3e00` |
| Rest of list | **pantheon-layout** — intentional Hermes-vs-Pantheon tree contract skips |
| Expected yaml shrink after fork test fix | **43 → 42** entries |
