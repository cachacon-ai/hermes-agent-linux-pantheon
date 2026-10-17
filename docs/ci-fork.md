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

**Sharding:** when `CI_TEST_SLICE_COUNT` is greater than 1 (default **4** on forks), `.github/workflows/tests.yml` runs the Linux pytest lane as parallel matrix shards. A lightweight `Generate test slices` job uses `test_durations.json` (Actions cache) and LPT balancing via `scripts/run_tests_parallel.py --generate-slices`. Each shard runs `scripts/run_tests.sh --files …` with the same env as the monolithic job. A final job named **Run tests** aggregates shard results so branch protection can keep requiring that check name. Set `CI_TEST_SLICE_COUNT=1` to restore a single job (upstream 96-core default).

## Workflows affected by runner variables

- `tests.yml`, `js-tests.yml`, `rust-tests.yml`, `nix.yml`, `e2e-desktop.yml` — `CI_LINUX_RUNNER`
- `tests-os.yml` — `CI_WINDOWS_RUNNER` on the Windows matrix leg
- `docker.yml` — `CI_LINUX_RUNNER` (amd64), `CI_LINUX_ARM_RUNNER` (arm64)

## Primary-repository automation

Jobs gated by `CI_PRIMARY_REPOSITORY` include Docker image build/publish, GitHub Pages docs deploy, and scheduled skills-index maintenance. Leave the variable **empty** on forks that should not push images or deploy Pages. Set it to the full repository name (e.g. `cachacon-ai/hermes-agent-linux-pantheon`) only when that repo owns the credentials and environments those jobs need.
