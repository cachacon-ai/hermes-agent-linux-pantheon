#!/usr/bin/env bash
# Prove a user on some earlier commit can reach this one.
#
# Installs a real, earlier Hermes the way a user does, applies ONE update route,
# and requires the checkout to land on this commit with a working `hermes`.
#
# Nothing here is mocked. scripts/dev-sandbox.sh provides the fake Internet --
# a bubblewrap sandbox with no writable host mounts, a MITM proxy serving the
# canonical install.sh URL, and a git-upload-pack shim standing in for
# github.com -- so `install.sh` really installs uv, a managed Python, Node and
# the venv, cloning "github.com" over the ssh-first path a user hits.
#
# One route per run, on a sandbox built from scratch, because the routes are only
# meaningful from a pristine install. Sharing one install across routes -- or
# rewinding the checkout with `git reset --hard` between them -- leaves the
# second route running against a tree the first already updated (same venv, same
# installed console script, same __pycache__), which is not the state any real
# user is in: a route can then pass only because its predecessor did the work,
# and a failure in the first leaves the second exercising something undefined.
# If you add a route, give it its own run.
#
# Usage:
#   tests/install/install-update-e2e.sh --route update|installer
#                                       [--install-ref REF] [--keep]
#                                       [--origin-url URL] [--backup-env NAME]
#                                       [--services] [--fork-surface]
#                                       [--rollback]
#
#   --route         which update path to exercise (required):
#                     update     `hermes update`
#                     installer  re-running the curl one-liner over the checkout
#   --install-ref   what to install first; anything git resolves (a branch, a
#                   tag like v2026.7.7, or a SHA reachable from main).
#                   Default: refs/heads/main.
#
# Optional phases (all off by default, so upstream's invocation is unchanged;
# the fork's workflow turns them on):
#   --origin-url    after the first install, point the checkout's `origin` at
#                   URL (e.g. the fork's git@github.com:owner/repo.git), the
#                   way a fork checkout is configured, so the update route
#                   runs `hermes update`'s fork path. The sandbox's ssh shim
#                   serves the fake remote for any github.com URL, so nothing
#                   leaves the sandbox. Asserted unchanged after the update.
#   --backup-env    before updating, create the local branch
#                   backup/<NAME>-pre-<target12>-<YYYYMMDD> at the installed
#                   commit (the production pre-upgrade snapshot practice) and
#                   assert it exists, points at the pre-upgrade commit, and is
#                   never pushed to the remote. Local to the sandbox only.
#   --services      after install and after the update, start the long-running
#                   services (`hermes gateway run` and the dashboard backend
#                   `hermes serve`) and require /api/health ok and
#                   /api/status gateway_running before stopping them.
#   --fork-surface  after the update, require the fork's own surface at the
#                   target: its modules import in the venv and its dashboard
#                   routes are mounted (HERMES_E2E_FORK_MODULES /
#                   HERMES_E2E_FORK_ROUTES override the lists).
#   --rollback      after the update passes, roll the checkout back to the
#                   pre-upgrade commit (via the --backup-env ref, else the
#                   installed SHA), reinstall its dependencies into the same
#                   venv, and require HEAD, `hermes --version` and (with
#                   --services) the service health checks to pass again.
#
# Requires a CLEAN worktree: every dev-sandbox invocation re-derives fake main
# from the working copy, so uncommitted changes move the update target between
# the call that installs and the call that verifies.

set -euo pipefail

ROUTE=""
INSTALL_REF="refs/heads/main"
KEEP=false
ORIGIN_URL=""
BACKUP_ENV=""
SERVICES=false
FORK_SURFACE=false
ROLLBACK=false
while [ "$#" -gt 0 ]; do
  case "$1" in
    --route)
      [ "$#" -ge 2 ] || { echo 'error: --route needs a value' >&2; exit 1; }
      ROUTE="$2"; shift 2 ;;
    --install-ref)
      [ "$#" -ge 2 ] || { echo 'error: --install-ref needs a value' >&2; exit 1; }
      INSTALL_REF="$2"; shift 2 ;;
    --keep) KEEP=true; shift ;;
    --origin-url)
      [ "$#" -ge 2 ] || { echo 'error: --origin-url needs a value' >&2; exit 1; }
      ORIGIN_URL="$2"; shift 2 ;;
    --backup-env)
      [ "$#" -ge 2 ] || { echo 'error: --backup-env needs a value' >&2; exit 1; }
      BACKUP_ENV="$2"; shift 2 ;;
    --services) SERVICES=true; shift ;;
    --fork-surface) FORK_SURFACE=true; shift ;;
    --rollback) ROLLBACK=true; shift ;;
    -h|--help) sed -n '2,70p' "$0"; exit 0 ;;
    *) echo "error: unknown argument: $1" >&2; exit 1 ;;
  esac
done
case "$ROUTE" in
  update|installer) ;;
  '') echo 'error: --route is required (update or installer)' >&2; exit 1 ;;
  *) echo "error: unknown route: $ROUTE (want update or installer)" >&2; exit 1 ;;
esac
case "$BACKUP_ENV" in
  ''|*[!A-Za-z0-9._-]*)
    [ -z "$BACKUP_ENV" ] || { echo "error: --backup-env must be [A-Za-z0-9._-]+: $BACKUP_ENV" >&2; exit 1; } ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# Keep sandbox state out of the default .hermes-sandbox so a run never clobbers
# a developer's own sandbox, and scope it per route so two routes can run
# concurrently (CI runs them as parallel matrix legs). dev-sandbox.sh joins this
# onto the worktree root and feeds it to `tar --exclude`, so it MUST be a
# relative directory name.
SANDBOX_DIR_NAME=".hermes-sandbox-e2e-$ROUTE"
export HERMES_DEV_SANDBOX_DIR="$SANDBOX_DIR_NAME"

SANDBOX_ROOT="$REPO_ROOT/$SANDBOX_DIR_NAME"
INSTALL_DIR="/home/hermes/.hermes/hermes-agent"   # user-level layout (sandbox default)
FAKE_REMOTE="/work/repos/hermes-agent.git"
# Only used to fetch an old install.sh for the flag probe below; the sandbox does
# its own fetching. Same override dev-sandbox.sh honours, so a fork can retarget
# both together.
UPSTREAM_URL="${HERMES_DEV_SANDBOX_UPSTREAM:-https://github.com/NousResearch/hermes-agent.git}"

# Installer transcripts live outside the sandbox root: the sandbox is recreated
# and (unless --keep) deleted, and these logs are the most useful artifact when
# a real install breaks. Created after the dirty check below, so that a log dir
# pointed inside the repo cannot be the thing that makes the tree dirty.
LOG_DIR="${HERMES_E2E_LOG_DIR:-$(mktemp -d -t hermes-install-e2e-logs.XXXXXX)}"

step() { printf '\n\033[1;36m▶ %s\033[0m\n' "$*"; }
ok()   { printf '\033[1;32m  ✓ %s\033[0m\n' "$*"; }
fail() { printf '\n\033[1;31m✗ %s\033[0m\n' "$*" >&2; exit 1; }

# The sandbox's internal logs (fake-internet proxy, slirp) explain failures that
# happen BEFORE install.sh gets to say anything -- a TLS handshake the proxy
# rejected looks like a bare `curl: (35)` from outside. Copy them out where a CI
# artifact upload can find them, and echo the proxy log since it is the usual
# culprit.
collect_sandbox_logs() {
  # Separate `local` statements on purpose: a single `local a=$1 b="$a"` does
  # NOT see the earlier assignment, so under `set -u` the second expansion dies
  # with "a: unbound variable".
  local tag="$1"
  local src="$SANDBOX_ROOT/root/logs"
  local dest="$LOG_DIR/sandbox-$tag"
  [ -d "$src" ] || return 0
  mkdir -p "$dest"
  cp -a "$src/." "$dest/" 2>/dev/null || true
  # Print it, not just archive it: a rejected TLS handshake here is the whole
  # explanation for a failure that otherwise reads as a bare `curl: (35)`, and
  # whoever is reading the job log should not have to download an artifact to
  # see it. In full, not tailed -- the file is short, and the useful line is not
  # reliably at the end.
  if [ -s "$dest/proxy.log" ]; then
    echo "--- sandbox proxy.log ---" >&2
    cat "$dest/proxy.log" >&2
    echo "--- end proxy.log ---" >&2
  fi
}

# ── preflight ──────────────────────────────────────────────────────────────
# Prefer the `sandbox` wrapper from the Nix devShell: it supplies both the PATH
# (bwrap, slirp4netns, openssl, ...) and the DEV_SANDBOX_* variables the script
# needs -- notably DEV_SANDBOX_DYNAMIC_LINKER, without which it cannot find a
# glibc loader on NixOS. Off Nix, the script is the entry point and finds its
# dependencies on the system PATH.
if command -v sandbox >/dev/null 2>&1; then
  SANDBOX=(sandbox)
elif command -v bwrap >/dev/null 2>&1; then
  SANDBOX=("$REPO_ROOT/scripts/dev-sandbox.sh")
else
  fail 'no usable sandbox: enter the Nix devShell (for `sandbox`) or install bubblewrap'
fi

if [ -n "$(git status --porcelain)" ]; then
  printf '\033[1;31m✗ working tree is dirty:\033[0m\n' >&2
  git status --porcelain | sed 's/^/    /' >&2
  fail 'Every sandbox invocation re-snapshots the working copy into a new
  fake-main commit, so the update target would move mid-run. Commit or stash
  first. (If a path above is build or log output, it needs gitignoring or to
  live outside the repo.)'
fi

mkdir -p "$LOG_DIR"

if [ "$KEEP" = false ]; then
  trap 'rm -rf -- "$SANDBOX_ROOT"' EXIT INT TERM
fi
rm -rf -- "$SANDBOX_ROOT"

# ── helpers ────────────────────────────────────────────────────────────────
# Does the INSTALLED hermes accept FLAG on `hermes update`?
#
# Asked of the installed binary rather than parsed out of a release's source:
# the update subcommand has lived in main.py, subcommands/update.py, and
# update_cmd.py across the releases we sample, so any static parse is a guess
# that silently rots. `hermes update --help` is the same surface a user meets,
# and argparse prints every option it accepts.
update_supports() {
  local flag="$1"
  in_sandbox "hermes update --help 2>&1" | grep -qF -- "$flag"
}

# Does the installer at REF accept FLAG? Read it out of that ref's own
# install.sh rather than assuming this checkout's flag set: the point of the
# matrix is to install releases from months back, whose installers predate
# options we take for granted. (Unlike the updater, the installer runs before
# anything is installed, so there is no --help to ask yet.)
#
# The ref may not be local -- the sandbox does its own fetching -- so fall back
# to fetching just that blob. Unresolvable means "flag absent", which costs a
# more conservative invocation, never a wrong one.
installer_supports() {
  local ref="$1"
  local flag="$2"
  local script=""
  script="$(git show "$ref:scripts/install.sh" 2>/dev/null)" || {
    git fetch -q --depth 1 "$UPSTREAM_URL" "$ref" 2>/dev/null || return 1
    script="$(git show FETCH_HEAD:scripts/install.sh 2>/dev/null)" || return 1
  }
  printf '%s' "$script" | grep -qF -- "$flag"
}

# Run the real install one-liner inside the sandbox. `ref` non-empty installs
# that upstream commit and promotes THIS checkout to fake main afterwards,
# leaving the state a user is in when an update is waiting; empty serves this
# worktree's own installer and points fake main here.
install_in_sandbox() {
  local what="$1"
  local ref="$2"
  local tag="$3"
  local log="$LOG_DIR/$tag.log"
  local args=(install --persistent)
  [ -n "$ref" ] && args+=(--install-ref "$ref")

  # Installer flags have to match the installer being run, not this checkout's.
  # Older releases reject options added later ("Unknown option: --skip-browser"),
  # and this test deliberately installs releases from months back. --skip-setup
  # goes back further than any tag we sample; anything newer is probed for.
  local installer_flags=(--skip-setup)
  if [ -z "$ref" ] || installer_supports "$ref" --skip-browser; then
    installer_flags+=(--skip-browser)
  fi
  # Sandbox flags must precede `--`; the rest goes to install.sh.
  args+=(-- "${installer_flags[@]}")

  # Stream the installer's output to stdout AND keep a copy on disk. It is the
  # substance of this test -- a real install of uv, a managed Python, Node and
  # the venv -- so it belongs in the job log where anyone reading the run can
  # see it, not only in an artifact they have to download. The file copy is what
  # the artifact upload keeps and what the failure paths grep.
  #
  # `set -o pipefail` is load-bearing here: without it the pipeline reports
  # tee's status and a failed install looks like a pass.
  local status=0
  "${SANDBOX[@]}" "${args[@]}" 2>&1 | tee "$log" || status=$?

  if [ "$status" -ne 0 ]; then
    collect_sandbox_logs "$tag"
    fail "$what failed (exit $status)"
  fi
  grep -q 'Installation Complete' "$log" \
    || { collect_sandbox_logs "$tag"; \
         fail "$what did not report a completed install"; }
  ok "$what completed (log: $log)"
}

in_sandbox() { "${SANDBOX[@]}" --persistent bash -lc "$1"; }

# fake main's SHA is read fresh whenever it is needed, never cached across a
# sandbox invocation: each invocation re-derives it from the worktree.
sandbox_target() { in_sandbox "git --git-dir=$FAKE_REMOTE rev-parse main" | tr -d '[:space:]'; }
sandbox_head()   { in_sandbox "cd $INSTALL_DIR && git rev-parse HEAD" | tr -d '[:space:]'; }

require_landed_on_target() {
  local what="$1" head target
  head="$(sandbox_head)"
  target="$(sandbox_target)"
  [ "$head" = "$target" ] || fail "$what left HEAD at $head, wanted $target"
  ok "$what landed on ${head:0:12}"
}

# The real smoke test: goes through the venv launcher and imports the app, so it
# fails if the venv, dependencies, or entry point are broken.
require_hermes_works() {
  local when="$1" out
  out="$(in_sandbox "hermes --version" 2>&1)" \
    || { printf '%s\n' "$out" >&2; fail "hermes --version failed $when"; }
  printf '%s\n' "$out" | sed 's/^/    /'
  ok "hermes runs $when"
}

# ── optional fork-aware phases ─────────────────────────────────────────────
# Default fork surface: modules and dashboard routes the fork added on top of
# upstream (Pantheon tool/artifacts, GPT-Live voice). Overridable so the list
# can follow the fork without editing this script.
FORK_MODULES="${HERMES_E2E_FORK_MODULES:-tools.pantheon_tool agent.pantheon_artifacts tools.voice_live}"
FORK_ROUTES="${HERMES_E2E_FORK_ROUTES:-/api/audio/voice-live/status}"
SERVICE_PORT="${HERMES_E2E_SERVICE_PORT:-9119}"

sandbox_origin() { in_sandbox "cd $INSTALL_DIR && git remote get-url origin" | tr -d '[:space:]'; }

set_origin() {
  [ -n "$ORIGIN_URL" ] || return 0
  in_sandbox "cd $INSTALL_DIR && git remote set-url origin '$ORIGIN_URL'" \
    || fail "could not point origin at $ORIGIN_URL"
  [ "$(sandbox_origin)" = "$ORIGIN_URL" ] || fail "origin is not $ORIGIN_URL after set-url"
  ok "origin -> $ORIGIN_URL"
}

require_origin_kept() {
  local what="$1" got
  [ -n "$ORIGIN_URL" ] || return 0
  got="$(sandbox_origin)"
  [ "$got" = "$ORIGIN_URL" ] || fail "$what rewrote origin to $got (wanted $ORIGIN_URL)"
  ok "$what kept origin at $ORIGIN_URL"
}

BACKUP_REF=""
create_backup_ref() {
  local base="$1" target="$2" got
  [ -n "$BACKUP_ENV" ] || return 0
  BACKUP_REF="backup/${BACKUP_ENV}-pre-${target:0:12}-$(date -u +%Y%m%d)"
  in_sandbox "cd $INSTALL_DIR && git branch '$BACKUP_REF' '$base'" \
    || fail "could not create backup ref $BACKUP_REF"
  require_backup_ref "$base" 'before the update'
}

require_backup_ref() {
  local base="$1" when="$2" got
  [ -n "$BACKUP_REF" ] || return 0
  got="$(in_sandbox "cd $INSTALL_DIR && git rev-parse --verify -q 'refs/heads/$BACKUP_REF^{commit}'" | tr -d '[:space:]')"
  [ "$got" = "$base" ] \
    || fail "backup ref $BACKUP_REF is '${got:-missing}' $when, wanted pre-upgrade $base"
  # Snapshot refs stay local: the remote must never learn about them.
  if in_sandbox "git --git-dir=$FAKE_REMOTE show-ref" | grep -qF "refs/heads/$BACKUP_REF"; then
    fail "backup ref $BACKUP_REF reached the remote $when"
  fi
  ok "backup ref $BACKUP_REF -> ${base:0:12} $when (local only)"
}

# Start the long-running services inside ONE sandbox invocation (each
# invocation is its own pid namespace, so anything backgrounded dies with it),
# wait for them to report healthy, then stop them. The gateway runs with no
# messaging platforms configured, which it supports (cron-only mode).
# Proxy env is set inside the sandbox, so curl must bypass it for loopback.
# $1: label, $2: "fork" to also require the fork's dashboard routes.
require_services_healthy() {
  local when="$1" surface="${2:-}" tag
  [ "$SERVICES" = true ] || return 0
  tag="services-$(printf '%s' "$when" | tr -c 'A-Za-z0-9' '-')"
  local routes=""
  [ "$surface" = fork ] && routes="$FORK_ROUTES"
  if ! in_sandbox "
    set -u
    port=$SERVICE_PORT
    logdir=/work/logs/$tag
    mkdir -p \"\$logdir\"
    hermes gateway run >\"\$logdir/gateway.log\" 2>&1 & gw=\$!
    hermes serve --host 127.0.0.1 --port \"\$port\" >\"\$logdir/serve.log\" 2>&1 & sv=\$!
    stop() { kill \$gw \$sv 2>/dev/null; wait \$gw \$sv 2>/dev/null || true; }
    dump() {
      echo '--- gateway.log (tail) ---'; tail -n 80 \"\$logdir/gateway.log\"
      echo '--- serve.log (tail) ---'; tail -n 80 \"\$logdir/serve.log\"
    }
    trap stop EXIT
    get() { curl -fsS --noproxy '*' --max-time 5 \"http://127.0.0.1:\$port\$1\"; }
    health=''
    for _ in \$(seq 1 180); do
      health=\$(get /api/health 2>/dev/null) && break
      health=''
      kill -0 \$sv 2>/dev/null || break
      sleep 1
    done
    [ -n \"\$health\" ] || { echo 'dashboard backend never answered /api/health'; dump; exit 1; }
    echo \"    /api/health: \$health\"
    printf '%s' \"\$health\" | python3 -c 'import json,sys; sys.exit(0 if json.load(sys.stdin).get(\"ok\") is True else 1)' \
      || { echo '/api/health did not report ok:true'; dump; exit 1; }
    running=''
    for _ in \$(seq 1 180); do
      status=\$(get /api/status 2>/dev/null) || status=''
      if [ -n \"\$status\" ] && printf '%s' \"\$status\" | python3 -c 'import json,sys; sys.exit(0 if json.load(sys.stdin).get(\"gateway_running\") is True else 1)'; then
        running=yes; break
      fi
      kill -0 \$gw 2>/dev/null || break
      sleep 1
    done
    [ -n \"\$running\" ] || { echo 'gateway never reported running via /api/status'; echo \"last /api/status: \$status\"; dump; exit 1; }
    printf '%s' \"\$status\" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(\"    /api/status: version=%s gateway_running=%s gateway_state=%s\" % (d.get(\"version\"), d.get(\"gateway_running\"), d.get(\"gateway_state\")))'
    for route in $routes; do
      code=\$(curl -sS --noproxy '*' --max-time 10 -o /dev/null -w '%{http_code}' \"http://127.0.0.1:\$port\$route\") || code=000
      case \"\$code\" in
        404|000|5??) echo \"fork route \$route answered \$code (not mounted / broken)\"; dump; exit 1 ;;
        *) echo \"    fork route \$route answered \$code (mounted)\" ;;
      esac
    done
  "; then
    collect_sandbox_logs "$tag"
    fail "services unhealthy $when"
  fi
  ok "gateway + dashboard backend healthy $when"
}

require_fork_surface() {
  local when="$1" mod
  [ "$FORK_SURFACE" = true ] || return 0
  for mod in $FORK_MODULES; do
    in_sandbox "cd $INSTALL_DIR && ./venv/bin/python -c 'import $mod'" \
      || fail "fork module $mod does not import $when"
  done
  ok "fork modules import $when: $FORK_MODULES"
}

# Roll back the way an operator does: reset to the snapshot, reinstall that
# commit's dependencies into the same venv with the managed uv (the command
# `hermes update` and install.sh's fallback tier use), relaunch.
rollback_to_base() {
  local base="$1" target ref head
  target="${BACKUP_REF:-$base}"
  step "ROLLBACK: $target (${base:0:12}) from the updated checkout"
  if ! in_sandbox "
    set -e
    cd $INSTALL_DIR
    git reset --hard '$target'
    uv=\"\${HERMES_HOME:-\$HOME/.hermes}/bin/uv\"
    [ -x \"\$uv\" ] || uv=\$(command -v uv)
    VIRTUAL_ENV=$INSTALL_DIR/venv \"\$uv\" pip install -e '.[all]'
  " 2>&1 | tee "$LOG_DIR/rollback.log"; then
    collect_sandbox_logs rollback
    fail "rollback to $target failed"
  fi
  head="$(sandbox_head)"
  [ "$head" = "$base" ] || fail "rollback left HEAD at $head, wanted $base"
  ok "rolled back to ${base:0:12}"
  require_hermes_works 'after rollback'
  require_services_healthy 'after rollback'
}

# ── install the earlier Hermes ─────────────────────────────────────────────
step "installing $INSTALL_REF from $UPSTREAM_URL (real curl | install.sh: uv, Python, Node, venv)"
install_in_sandbox "install of $INSTALL_REF" "$INSTALL_REF" install

BASE="$(sandbox_head)"
TARGET="$(sandbox_target)"
[ -n "$BASE" ] || fail "could not read the installed commit"
[ "$BASE" != "$TARGET" ] \
  || fail "install landed on the update target ($BASE); base and target must differ"
ok "installed ${BASE:0:12}; update target is ${TARGET:0:12}"
require_hermes_works 'after install'
set_origin
require_services_healthy 'after install'
create_backup_ref "$BASE" "$TARGET"

# ── apply exactly one update route ─────────────────────────────────────────
case "$ROUTE" in
  update)
    step 'ROUTE: hermes update'
    # `--yes` reaches the update subcommand only in later releases, and argparse
    # rejects the whole invocation when it does not exist. Ask the installed
    # hermes which it accepts; older ones read the prompt from stdin, so close it.
    if update_supports --yes; then
      update_cmd="hermes update --yes"
    else
      update_cmd="hermes update </dev/null"
    fi
    if ! in_sandbox "cd $INSTALL_DIR && $update_cmd"; then
      collect_sandbox_logs update
      fail "hermes update failed ($update_cmd)"
    fi
    require_landed_on_target 'hermes update'
    require_hermes_works 'after hermes update'
    require_origin_kept 'hermes update'
    ;;
  installer)
    step 'ROUTE: installer re-run over the existing checkout'
    # No ref: serves this worktree's installer and points fake main at this
    # checkout, which is what the re-run must land on.
    install_in_sandbox 'installer re-run' '' reinstall
    require_landed_on_target 'installer re-run'
    require_hermes_works 'after installer re-run'
    require_origin_kept 'installer re-run'
    ;;
esac

require_backup_ref "$BASE" "after $ROUTE"
require_fork_surface "after $ROUTE"
fork_routes=""
[ "$FORK_SURFACE" = true ] && fork_routes=fork
require_services_healthy "after $ROUTE" "$fork_routes"

if [ "$ROLLBACK" = true ]; then
  rollback_to_base "$BASE"
fi

printf '\n\033[1;32m✓ install/update E2E passed (route: %s, from: %s%s)\033[0m\n' \
  "$ROUTE" "$INSTALL_REF" "$([ "$ROLLBACK" = true ] && echo ', rollback ok')"
[ "$KEEP" = true ] && echo "  sandbox kept at $SANDBOX_ROOT"
exit 0
