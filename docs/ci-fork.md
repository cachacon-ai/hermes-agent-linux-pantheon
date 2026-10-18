# Fork and personal-account CI

Upstream Hermes CI assumes Nous Research org-hosted **larger runners** (`ubuntu-latest-96-core`, `ubuntu-latest-32-core`, etc.) and automation that only runs on `NousResearch/hermes-agent`. Personal or fork copies should use **repository variables** so workflows stay mergeable with upstream without editing hardcoded labels on every sync.

## Repository variables

| Variable | When unset | Typical fork value | Upstream (Nous) value |
|----------|------------|--------------------|------------------------|
| `CI_LINUX_RUNNER` | `ubuntu-latest` | `ubuntu-latest` (default) | `ubuntu-latest-96-core` or `ubuntu-latest-32-core` per job |
| `CI_LINUX_ARM_RUNNER` | `ubuntu-latest` | Leave unset unless you run Docker arm64 builds | `ubuntu-latest-32-arm-core` |
| `CI_WINDOWS_RUNNER` | `windows-latest` | `windows-latest` | `windows-latest-32-core` |
| `CI_TEST_WORKERS` | `4` | `4` (matches standard 4 vCPU Linux runner) | `96` on the 96-core test runner |
| `CI_TEST_TIMEOUT_MINUTES` | `120` | `120` or higher if the suite is still tight | `30` on the 96-core runner |
| `CI_TEST_SLICE_COUNT` | `4` | `4` on standard `ubuntu-latest` (parallel shards) | `1` (single 96-core job, no matrix) |
| `CI_PRIMARY_REPOSITORY` | automation **off** | Set to `owner/repo` only when you want publish/deploy/skills automation on that repo | `NousResearch/hermes-agent` |
| `CI_PYTHON_MODE` | PRs: `pantheon`; push: `full` (see below) | `pantheon` | `full` (or unset — upstream runs the whole suite) |
| `CI_PYTHON_TEST_PATHS` | ignored unless `CI_PYTHON_MODE=paths` | Newline or comma-separated roots/files for `scripts/run_tests.sh` | — |
| `CI_PYTHON_EXTRA_ARGS` | ignored | Optional extra args forwarded to `run_tests.sh` (e.g. `-k pantheon`) | — |
| `CI_PYTHON_FULL_CRON` | `0 6 * * *` in workflow | Document only — edit `.github/workflows/tests.yml` `schedule` to match | — |

Set variables under **Settings → Secrets and variables → Actions → Variables**.

## Python test scope (`CI_PYTHON_MODE`)

The main pytest job (`.github/workflows/tests.yml`) invokes `scripts/ci/run_python_tests_ci.sh`, which reads `CI_PYTHON_MODE`:

| Mode | Behavior |
|------|----------|
| `full` | `scripts/run_tests.sh` with no path filter (entire suite). |
| `pantheon` | Only paths under `tests/` listed in `scripts/ci/pantheon_python_test_paths.txt`. |
| `paths` | Uses `CI_PYTHON_TEST_PATHS` (comma- or newline-separated). |

**Defaults when `CI_PYTHON_MODE` is unset:**

- `pull_request` events → `pantheon` (fast fork PRs).
- Other events (e.g. `push` to `main`) → `full`.

**Scheduled and manual full runs:** `tests.yml` also defines a daily `schedule` (06:00 UTC) and `workflow_dispatch` with `python_mode` defaulting to `full`. Both run the same job definition as CI PRs, with `CI_PYTHON_MODE=full`.

**Classifier widening:** `scripts/ci/classify_changes.py` exposes `pantheon_python_scope`. When a PR changes Python files *outside* the product + test prefixes in `pantheon_python_test_paths.txt`, the orchestrator passes `python_mode: full` into the tests workflow so shared-core edits are not validated by the Pantheon slice alone.

PRs that touch only `website/` (and other `_PY_SKIP` trees) already skip the Python lane via the existing `python` classifier gate.

## Python test job timing

The main pytest lane (`.github/workflows/tests.yml`) defaults to **120 minutes** timeout and **4** parallel test-file workers on standard `ubuntu-latest`. Upstream runs the same workflow with `CI_LINUX_RUNNER=ubuntu-latest-96-core`, `CI_TEST_WORKERS=96`, and `CI_TEST_TIMEOUT_MINUTES=30`.

If a fork run times out, raise `CI_TEST_TIMEOUT_MINUTES` (for example `180`) or lower `CI_TEST_WORKERS` to reduce contention.

**Sharding:** when `CI_TEST_SLICE_COUNT` is greater than 1 (default **4** on forks), `.github/workflows/tests.yml` runs the Linux pytest lane as parallel matrix shards. A lightweight `Generate test slices` job uses `test_durations.json` (Actions cache) and LPT balancing via `scripts/run_tests_parallel.py --generate-slices` (empty LPT buckets are omitted from the matrix). Each shard runs `scripts/run_tests.sh --files …` with the same env as the monolithic job. Exactly one GitHub check is named **Run tests**: the **`test-gate`** job, which passes when either the monolithic **`test`** job (`CI_TEST_SLICE_COUNT=1`, job name **Tests (single runner)**) or all **`test-shard`** matrix legs succeed. Sharded runs skip **Tests (single runner)** so branch protection never sees a skipped **Run tests**. An explicit empty `--files` list exits successfully without re-discovering the full suite.

## Workflows affected by runner variables

- `tests.yml`, `js-tests.yml`, `rust-tests.yml`, `nix.yml`, `e2e-desktop.yml` — `CI_LINUX_RUNNER`
- `tests-os.yml` — `CI_WINDOWS_RUNNER` on the Windows matrix leg
- `docker.yml` — `CI_LINUX_RUNNER` (amd64), `CI_LINUX_ARM_RUNNER` (arm64)

## Primary-repository automation

Jobs gated by `CI_PRIMARY_REPOSITORY` include Docker image build/publish, GitHub Pages docs deploy, and scheduled skills-index maintenance. Leave the variable **empty** on forks that should not push images or deploy Pages. Set it to the full repository name (e.g. `cachacon-ai/hermes-agent-linux-pantheon`) only when that repo owns the credentials and environments those jobs need.

## Install & Update E2E (fork-native)

`.github/workflows/install-e2e.yml` (unchanged from upstream) proves a real `curl | install.sh` install of an earlier version can update to the tested commit. Upstream picks `vYYYY.M.D` release tags, which this fork does not carry, so every run failed in **Pick release tags**. The fork-native behaviour lives in the scripts the workflow calls:

- `scripts/sandbox/pick-release-tags.sh` keeps upstream's `vYYYY.M.D` scheme as its default, and switches to the fork's own scheme, `known-good-linux-pantheon-YYYY-MM-DD`, when the repo carries `tests/install/fork-e2e.conf` (checked in HEAD's tree too, so CI's sparse checkout sees it). `HERMES_E2E_TAG_PATTERN` or `--pattern` override both; `--merged REF` restricts picks to ancestors of REF.
- `tests/install/fork-e2e.conf`, when present, makes `tests/install/install-update-e2e.sh` run fork-native (`--no-fork` restores upstream behaviour).

Each leg (route `update` = `hermes update`, route `installer` = re-running the installer), all inside the bubblewrap sandbox on the runner:

1. Installs the tag, fetched from this repository rather than `NousResearch/hermes-agent`; checks `hermes --version`.
2. Points the checkout's `origin` at this fork (as a fork checkout is configured), then starts `hermes gateway run` + `hermes serve` and requires `/api/health` `ok: true` and `/api/status` `gateway_running: true`.
3. Creates the local snapshot branch `backup/ci-e2e-pre-<target12>-<YYYYMMDD>` at the installed commit and asserts it points there and never reaches the remote.
4. Applies the route; requires HEAD = tested commit, origin unchanged, snapshot intact, the fork's modules (`tools.pantheon_tool`, `agent.pantheon_artifacts`, `tools.voice_live`) importable, the fork's dashboard route (`/api/audio/voice-live/status`) mounted, and the services healthy.
5. Rolls back to the snapshot (`git reset --hard` + `uv pip install -e '.[all]'` into the same venv) and requires HEAD = tag, `hermes --version`, and healthy services again.

Nothing is deployed, no production host or secret is used, and no ref is pushed. To add a new upgrade baseline, push a tag following the scheme (e.g. `known-good-linux-pantheon-2026-11-01`) on a commit in `main`'s history; the scheduled runs pick it up.
