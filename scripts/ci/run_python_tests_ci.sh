#!/usr/bin/env bash
# CI entrypoint for scripts/run_tests.sh — honors CI_PYTHON_MODE and repo vars.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=python_test_ci_lib.sh
source "$SCRIPT_DIR/python_test_ci_lib.sh"

REPO_ROOT="$(python_test_ci_repo_root)"

extra_args=()
if [[ -n "${CI_PYTHON_EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  extra_args=(${CI_PYTHON_EXTRA_ARGS})
fi

python_test_ci_build_path_args
mode="$(python_test_ci_resolve_mode)"

case "$mode" in
  full)
    exec "$REPO_ROOT/scripts/run_tests.sh" "${extra_args[@]}"
    ;;
  pantheon | paths)
    exec "$REPO_ROOT/scripts/run_tests.sh" "${PYTHON_TEST_PATH_ARGS[@]}" "${extra_args[@]}"
    ;;
  *)
    echo "Unknown CI_PYTHON_MODE: $mode (expected full, pantheon, or paths)" >&2
    exit 1
    ;;
esac
