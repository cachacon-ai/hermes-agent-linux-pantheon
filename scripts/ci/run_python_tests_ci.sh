#!/usr/bin/env bash
# CI entrypoint for scripts/run_tests.sh — honors CI_PYTHON_MODE and repo vars.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PATHS_FILE="$SCRIPT_DIR/pantheon_python_test_paths.txt"

mode="${CI_PYTHON_MODE:-}"
if [[ -z "$mode" ]]; then
  if [[ "${GITHUB_EVENT_NAME:-}" == "pull_request" ]]; then
    mode="pantheon"
  else
    mode="full"
  fi
fi

extra_args=()
if [[ -n "${CI_PYTHON_EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  extra_args=(${CI_PYTHON_EXTRA_ARGS})
fi

read_paths_file() {
  local line
  while IFS= read -r line || [[ -n "$line" ]]; do
    line="${line%%#*}"
    line="${line#"${line%%[![:space:]]*}"}"
    line="${line%"${line##*[![:space:]]}"}"
    [[ -z "$line" ]] && continue
    printf '%s\n' "$line"
  done < "$PATHS_FILE"
}

case "$mode" in
  full)
    exec "$REPO_ROOT/scripts/run_tests.sh" "${extra_args[@]}"
    ;;
  pantheon)
    mapfile -t pantheon_paths < <(read_paths_file | grep '^tests/' || true)
    if [[ ${#pantheon_paths[@]} -eq 0 ]]; then
      echo "No tests/ entries in pantheon_python_test_paths.txt" >&2
      exit 1
    fi
    exec "$REPO_ROOT/scripts/run_tests.sh" "${pantheon_paths[@]}" "${extra_args[@]}"
    ;;
  paths)
    if [[ -z "${CI_PYTHON_TEST_PATHS:-}" ]]; then
      echo "CI_PYTHON_MODE=paths requires CI_PYTHON_TEST_PATHS" >&2
      exit 1
    fi
    mapfile -t custom_paths < <(printf '%s\n' "${CI_PYTHON_TEST_PATHS}" | tr ',' '\n')
    trimmed=()
    for p in "${custom_paths[@]}"; do
      p="${p#"${p%%[![:space:]]*}"}"
      p="${p%"${p##*[![:space:]]}"}"
      [[ -z "$p" ]] && continue
      trimmed+=("$p")
    done
    exec "$REPO_ROOT/scripts/run_tests.sh" "${trimmed[@]}" "${extra_args[@]}"
    ;;
  *)
    echo "Unknown CI_PYTHON_MODE: $mode (expected full, pantheon, or paths)" >&2
    exit 1
    ;;
esac
