# GPT-Live ChatGPT subscription auth — port spec (Hermes fork → Pantheon)

**Investigation only.** Base: fork `main` @ `80c08658c6`. Upstream reference: NousResearch/hermes-agent PR [#136326](https://github.com/NousResearch/hermes-agent/pull/136326) (`706cb12`, open). Related: [#132588](https://github.com/NousResearch/hermes-agent/pull/132588) (Codex token on `/v1/live/sessions` — **not** equivalent to subscription quicksilver), [#108940](https://github.com/NousResearch/hermes-agent/issues/108940) (conflicted predecessor), community [hermes-talk](https://github.com/TheSmokeDev/hermes-talk) v0.21.1 (same quicksilver URL; extra headers).

**Goal:** Chris uses Pantheon iPhone → Hermes on BeyondMax with `voice.gpt_live.auth: subscription` (Codex OAuth / ChatGPT plan), not `OPENAI_API_KEY`.

---

## 0. Workflow order (non-negotiable)

1. **Hermes fork** (`cachacon-ai/hermes-agent-linux-pantheon` or equivalent): implement + test subscription GPT-Live here first.
2. **Bump `HERMES_RUNTIME_PIN`** in Pantheon (`scripts/sync-hermes-runtime.sh`, `hermes-runtime-pin.yml` CI).
3. **Pantheon app/service**: adapt to new HTTP + WebRTC event shapes; run Pantheon vitest/e2e.
4. Run **`scripts/ci/classify_changes.py`**: any change under `scripts/ci/pantheon_python_test_paths.txt` widens Pantheon PR pytest (includes `tools/voice_live.py`, `hermes_cli/web_server.py`, `hermes_cli/config_defaults.py`, `tests/hermes_cli/test_voice_live_routes.py`, `tests/tui_gateway/test_voice_live_delegation.py`).

---

## 1. Port PR #136326 — server side into this fork

Upstream touches **25 files**; **Pantheon-relevant server subset** below. Desktop/i18n files are N/A for iOS-only Pantheon but document delegation wire differences Pantheon must mirror.

### 1.1 `tools/voice_live.py` (primary)

| Area | Fork today (`80c08658c6`) | Port from #136326 | Merge notes |
|------|-------------------------|-------------------|-------------|
| Auth modes | API key only via `_resolve_credentials()` | `voice.gpt_live.auth`: `api` \| `subscription` (default `api`) | Keep fork **fail-closed** error types (`VoiceLiveUpstreamError`, no upstream body in client/logs) for API path; map subscription failures to `RuntimeError`/`ValueError` like upstream **or** extend `VoiceLiveUpstreamError` without leaking bodies |
| Subscription endpoint | — | `POST https://chatgpt.com/backend-api/codex/realtime/calls?intent=quicksilver&architecture=avas` | JSON body `{"sdp", "session"}` — **no** `transport` wrapper |
| Headers | `Authorization: Bearer <api_key>` | `Authorization`, `ChatGPT-Account-Id`, `OpenAI-Alpha: quicksilver=v2` | hermes-talk also sends `session-id`, `thread-id`, `x-session-id` (UUIDs); **probe both** if 403 — not in #136326 tests |
| Models/voices | `gpt-live-1` / `marin` + `GPT_LIVE_VOICES` | `subscription_model` default `gpt-live-1-codex`, `subscription_voice` default `cove` (+ curated list in desktop settings) | Pantheon config only needs `subscription_*` when `auth: subscription` |
| History key | `input` (normalized via `normalize_session_history`) | `initial_items` for subscription, `input` for API | **Must** run fork normalizer then assign to correct key |
| Session create | `create_webrtc_session` → urllib `POST …/v1/live/sessions` | Branch: `_create_subscription_session` (httpx, `follow_redirects=False`) | Keep `ensure_gpt_live_mode()` **before** network (fork #29) |
| Response shape | Vendor JSON `{session, transport}` | Subscription: **raw SDP text** body + `Location` header → synthetic `{auth, session, transport}` | Add `"auth": "api"` \| `"subscription"` on success (upstream) |
| `expected_auth` | — | Pin status preflight to session create | Prevents billing mode flip between GET status and POST session |
| Status | `available` + `reason` internally; route strips to `configured` | `auth`, `available`, `reason`, `model`, `voice` | **Fork/Pantheon conflict** — see §2 |
| Hangup | `close_live_session()` → `POST /v1/live/sessions/{id}/hangup` (**#30/#31**) | **Not in #136326** | Subscription `rtc_*` ids are **not** on OpenAI `/v1/live/sessions/.../hangup`; need separate teardown spec (§2.4) |

**`auth_codex` vs `auth.py`:** Upstream PR imports `hermes_cli.auth_codex.resolve_codex_runtime_credentials` and `auth_constants.AuthError`, with `read_only=True` on status. **This fork has no `auth_codex.py`** — re-point to:

- `from hermes_cli.auth import resolve_codex_runtime_credentials, AuthError, _decode_jwt_claims`

Add **`read_only: bool = False`** to `resolve_codex_runtime_credentials()` (mirror upstream `706cb12` behavior): when `read_only=True`, do not refresh or mutate `auth.json` on status checks.

**#132588 refutation:** That PR makes **API** `/v1/live/sessions` accept a Codex OAuth bearer from the pool. It does **not** implement quicksilver subscription billing, `gpt-live-1-codex`, or `initial_items` / frameless events. Useful as a **probe fallback** only; not a substitute for #136326 subscription mode.

### 1.2 `hermes_cli/config_defaults.py`

Add under `voice.gpt_live`:

```yaml
auth: api
subscription_model: gpt-live-1-codex
subscription_voice: cove
```

(Keep existing `model`, `voice`, `instructions`, optional `api_key` / `base_url`.)

### 1.3 `hermes_cli/web_models.py`

Add to `VoiceLiveSessionRequest`:

```python
expected_auth: Optional[Literal["api", "subscription"]] = None
```

### 1.4 HTTP routes — **fork divergence (#29)**

Upstream: `hermes_cli/web_routers/audio.py`. **Fork:** routes live in `hermes_cli/web_server.py` ~5541–5673.

| Route | Upstream #136326 | Fork today | Required change |
|-------|------------------|------------|-----------------|
| `GET /api/audio/voice-live/status` | `return {"ok": True, **result}` | Maps `available`→`configured`, drops `reason`, no `auth` | **Decision:** either (A) pass-through upstream fields for Pantheon parity, or (B) keep `configured` and add `auth`, `reason` aliases — Pantheon client must match |
| `POST /api/audio/voice-live/session` | `ValueError`→503, `RuntimeError`→502, passes `expected_auth` | Typed exceptions, 502 generic detail | Wire `expected_auth`; map subscription `RuntimeError` to 502 without vendor body |
| `POST /api/audio/voice-live/session/close` | *(unchanged in PR)* | Fork-only (#30) | Extend `close_live_session(session_id, auth=…)` for subscription teardown when known |

### 1.5 Tests to add/adapt

| File | Action |
|------|--------|
| `tests/tools/test_voice_live_subscription.py` | **Port** from PR; change `auth_codex` → `hermes_cli.auth` |
| `tests/hermes_cli/test_voice_live_routes.py` | Add subscription status/session cases; preserve fork invariants (no key leak, `configured` contract if kept, profile scoping, hangup #31 404/410) |
| `tests/tui_gateway/test_voice_live_delegation.py` | Unchanged for HTTP auth; still validates `surface=voice-live` delegation — no change unless event names documented for Pantheon service |

**CI:** All of the above paths are in `scripts/ci/pantheon_python_test_paths.txt` → Pantheon-mode PR CI runs them.

### 1.6 Docs / examples

- Port `cli-config.yaml.example` gpt_live comment block from PR.
- Optional: `website/docs/user-guide/features/voice-mode.md` (not required for Pantheon).

### 1.7 Dependency

#136326 adds `httpx` usage in `voice_live.py`. Fork already pins `httpx[socks]==0.28.1` in `pyproject.toml` — no new dep.

### 1.8 Explicit non-goals in #136326 (do not port for Pantheon)

- `apps/desktop/*` (15 files) — reference for WebRTC event normalization only.
- `contributors/emails/*` — skip.

---

## 2. Response shapes vs Pantheon iPhone client

Pantheon repo (`cachacon-ai/pantheon`, `pantheon-v1`) is not readable from this environment (private). Spec below is from fork HTTP contract + #136326 desktop client + user-cited `gpt_live_voice_session.dart` / PAN-49.

### 2.1 REST: `GET /api/audio/voice-live/status`

| Field | Fork today | After subscription port (upstream) | Pantheon action |
|-------|------------|-------------------------------------|-----------------|
| `ok` | `true` | `true` | unchanged |
| `mode` | `chained` \| `gpt-live` | same | unchanged |
| `configured` | `bool` (from `available`) | N/A upstream | **Keep** for backward compat OR dual-publish `available` |
| `auth` | **missing** | `api` \| `subscription` | **Parse required** before starting WebRTC |
| `reason` | **stripped** | string \| null when unavailable | Show error for subscription (no silent API fallback) |
| `model`, `voice` | present | mode-specific (`gpt-live-1-codex` / `cove` for subscription) | Use for UI labels; do not hardcode `gpt-live-1` |

**Chris config (BeyondMax `~/.hermes/config.yaml`):**

```yaml
voice:
  voice_chat_mode: gpt-live
  gpt_live:
    auth: subscription
    subscription_voice: cove   # optional; default cove
```

### 2.2 REST: `POST /api/audio/voice-live/session`

Request:

```json
{
  "sdp": "<WebRTC offer, byte-exact>",
  "history": [ … ],
  "expected_auth": "subscription"
}
```

Response (success):

```json
{
  "ok": true,
  "auth": "subscription",
  "session": { "id": "rtc_…" },
  "transport": { "type": "webrtc", "sdp": "v=0…" }
}
```

Fork already returns vendor fields at top level — adding `auth` is backward compatible.

### 2.3 WebRTC data channel — **delegation event split** (critical)

| API mode (`auth: api`) | Subscription quicksilver (`auth: subscription`) |
|------------------------|--------------------------------------------------|
| `session.delegation.created` + `delegation: {id,type,target}` | `delegation.created` + `item: {id,type,target,content?}` |
| `session.input_transcript.delta` / `session.output_transcript.delta` | `input_transcript.added` / `output_transcript.added` with `item.text` |
| Reply: `session.commentary.append` (+ `delegation_id`) | Reply: `delegation.context.append` with `channel: speakable`, chunks ≤500 UTF-8 bytes; thinking via `session.context.append` `channel: commentary` |

Pantheon Flutter (`gpt_live_voice_session.dart` / `bot_voice.dart`) must implement the **normalization layer** equivalent to upstream `apps/desktop/src/lib/voice-live.ts` (`normalizeSubscriptionEvent`, `subscriptionChunks`, dedupe `delegationIds`).

Hermes **gateway** side (`tui_gateway`, `surface=voice-live`) unchanged: still `prompt.submit` with voice transcript context; only the **iPhone WebRTC client** must speak the quicksilver dialect.

### 2.4 Close / hangup

| Auth | Teardown |
|------|----------|
| `api` | `POST /api/audio/voice-live/session/close` with `session_id` → fork `close_live_session` → OpenAI `POST /v1/live/sessions/{id}/hangup` (#30, 404/410 → `already_closed` #31) |
| `subscription` | **Undefined in #136326.** Likely WebRTC/datachannel `session.close` only. Optional future: Codex `DELETE /backend-api/codex/realtime/calls/{id}` (undocumented — probe script tries candidates). **Do not** call API hangup with `rtc_*` id expecting success. |

Pantheon: on disconnect, call Hermes close route only when `auth == api`; for subscription, close peer connection locally and optionally add fork support once endpoint is confirmed.

### 2.5 Pantheon service + pin bump

1. Implement fork changes; green `tests/tools/test_voice_live_subscription.py` + `test_voice_live_routes.py`.
2. Commit SHA → Pantheon `HERMES_RUNTIME_PIN` + `scripts/sync-hermes-runtime.sh`.
3. Flutter: status/auth handling, `expected_auth` on session POST, subscription event normalization, voice list for settings (cove, arbor, breeze, ember, juniper, maple, sol, spruce, vale).
4. If Pantheon microservice proxies voice-live, update its OpenAPI/types in lockstep.

---

## 3. Probe script

`scripts/probe_gpt_live_subscription_auth.py` — read-only step-0 on BeyondMax:

- Reads `~/.hermes/auth.json` (`HERMES_HOME`) Codex tokens.
- Refreshes in memory if expiring (skew 120s); **does not write** `auth.json` unless `--write-refreshed-tokens`.
- **Default:** subscription quicksilver POST with minimal audio SDP.
- **`--variant api-live-sessions`:** #132588 check — Codex bearer on `POST https://api.openai.com/v1/live/sessions` with API-shaped body.
- Redacts secrets in output; attempts best-effort hangup when call id returned.

---

## 4. Risks and size

| Risk | Impact |
|------|--------|
| Undocumented quicksilver endpoint / header drift | 403/401 until headers match Codex CLI (OpenClaw reference) |
| Plan gating (Plus/Pro/team) | Auth succeeds but 403 on call create — probe distinguishes |
| ToS / automation | Personal use on owned hardware; subscription route is ChatGPT-authenticated, not API key resale |
| Delegation parity | Wrong event names → voice connects but Hermes never gets turns |
| #136326 churn | 25-file PR may rebase; fork should port **behavior** not blind diff |
| #132588 confusion | Codex on `/v1/live/sessions` ≠ subscription billing path |
| Hangup gap | Leaked `rtc_*` calls if only WebRTC dropped |
| Multiplex profiles | `resolve_codex_runtime_credentials` must stay fail-closed under `?profile=` (see #132588 review on pool peek) |

**Size estimate (implementation, not calendar):**

| Repo | Scope | ~LOC / files |
|------|-------|----------------|
| Hermes fork | `voice_live.py`, `auth.py` read_only, config, web_models, web_server routes, tests, probe | **+400–550** LOC, **6–8** files |
| Pantheon Flutter | voice session + settings + types | **+250–400** LOC, **3–5** files |
| Pantheon pin/CI | pin bump, sync script | **1–2** files |

---

## 5. Line anchors (fork `80c08658c6`)

- API-only credentials: `tools/voice_live.py` `_resolve_credentials` L185–194, `create_webrtc_session` L346–382
- Hangup (API only today): `close_live_session` L403–461; route `hermes_cli/web_server.py` L5624–5673
- Status shaping: `resolve_gpt_live_status` L202–215; route strips to `configured` L5551–5561
- Delegation backend: `tui_gateway/methods_prompt.py` (`voice-live` surface), tests `tests/tui_gateway/test_voice_live_delegation.py`
- Pantheon CI scope: `scripts/ci/pantheon_python_test_paths.txt` L8–13, L36
