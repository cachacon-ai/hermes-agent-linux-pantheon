#!/usr/bin/env bash
# Emit a GitHub Actions matrix JSON for LPT-balanced test shards.
set -euo pipefail

slice_count="${1:?usage: generate_python_test_slices.sh <slice_count>}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=python_test_ci_lib.sh
source "$SCRIPT_DIR/python_test_ci_lib.sh"

REPO_ROOT="$(python_test_ci_repo_root)"
python_test_ci_build_path_args

cmd=(python3 "$REPO_ROOT/scripts/run_tests_parallel.py")
if [[ ${#PYTHON_TEST_PATH_ARGS[@]} -gt 0 ]]; then
  cmd+=("${PYTHON_TEST_PATH_ARGS[@]}")
fi
cmd+=(--generate-slices "$slice_count")

exec "${cmd[@]}"
