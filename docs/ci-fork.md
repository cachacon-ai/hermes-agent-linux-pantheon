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
| `CI_PRIMARY_REPOSITORY` | automation **off** | Set to `owner/repo` only when you want publish/deploy/skills automation on that repo | `NousResearch/hermes-agent` |

Set variables under **Settings → Secrets and variables → Actions → Variables**.

## Python test job timing

The main pytest lane (`.github/workflows/tests.yml`) defaults to **120 minutes** timeout and **4** parallel test-file workers on standard `ubuntu-latest`. Upstream runs the same workflow with `CI_LINUX_RUNNER=ubuntu-latest-96-core`, `CI_TEST_WORKERS=96`, and `CI_TEST_TIMEOUT_MINUTES=30`.

If a fork run times out, raise `CI_TEST_TIMEOUT_MINUTES` (for example `180`) or lower `CI_TEST_WORKERS` to reduce contention. There is no matrix sharding in the current workflow; adding sharding would be a separate change.

## Workflows affected by runner variables

- `tests.yml`, `js-tests.yml`, `rust-tests.yml`, `nix.yml`, `e2e-desktop.yml` — `CI_LINUX_RUNNER`
- `tests-os.yml` — `CI_WINDOWS_RUNNER` on the Windows matrix leg
- `docker.yml` — `CI_LINUX_RUNNER` (amd64), `CI_LINUX_ARM_RUNNER` (arm64)

## Primary-repository automation

Jobs gated by `CI_PRIMARY_REPOSITORY` include Docker image build/publish, GitHub Pages docs deploy, and scheduled skills-index maintenance. Leave the variable **empty** on forks that should not push images or deploy Pages. Set it to the full repository name (e.g. `cachacon-ai/hermes-agent-linux-pantheon`) only when that repo owns the credentials and environments those jobs need.
