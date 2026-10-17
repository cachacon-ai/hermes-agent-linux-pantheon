#!/usr/bin/env bash
# Shared CI helpers for Python test path/mode resolution (sourced, not executed).
# Used by run_python_tests_ci.sh and generate_python_test_slices.sh.

python_test_ci_repo_root() {
  local script_dir
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  cd "$script_dir/../.." && pwd
}

python_test_ci_resolve_mode() {
  local mode="${CI_PYTHON_MODE:-}"
  if [[ -z "$mode" ]]; then
    if [[ "${GITHUB_EVENT_NAME:-}" == "pull_request" ]]; then
      mode="pantheon"
    else
      mode="full"
    fi
  fi
  printf '%s' "$mode"
}

# Populates PYTHON_TEST_PATH_ARGS with discovery roots/files for run_tests_parallel.py.
# Full mode leaves the array empty (default discovery under tests/).
python_test_ci_build_path_args() {
  PYTHON_TEST_PATH_ARGS=()
  local mode paths_file repo_root line
  mode="$(python_test_ci_resolve_mode)"
  repo_root="$(python_test_ci_repo_root)"
  paths_file="$repo_root/scripts/ci/pantheon_python_test_paths.txt"

  read_paths_file() {
    while IFS= read -r line || [[ -n "$line" ]]; do
      line="${line%%#*}"
      line="${line#"${line%%[![:space:]]*}"}"
      line="${line%"${line##*[![:space:]]}"}"
      [[ -z "$line" ]] && continue
      printf '%s\n' "$line"
    done < "$paths_file"
  }

  case "$mode" in
    full)
      ;;
    pantheon)
      while IFS= read -r line; do
        PYTHON_TEST_PATH_ARGS+=("$line")
      done < <(read_paths_file | grep '^tests/' || true)
      if [[ ${#PYTHON_TEST_PATH_ARGS[@]} -eq 0 ]]; then
        echo "No tests/ entries in pantheon_python_test_paths.txt" >&2
        return 1
      fi
      ;;
    paths)
      if [[ -z "${CI_PYTHON_TEST_PATHS:-}" ]]; then
        echo "CI_PYTHON_MODE=paths requires CI_PYTHON_TEST_PATHS" >&2
        return 1
      fi
      local p
      while IFS= read -r p; do
        p="${p#"${p%%[![:space:]]*}"}"
        p="${p%"${p##*[![:space:]]}"}"
        [[ -z "$p" ]] && continue
        PYTHON_TEST_PATH_ARGS+=("$p")
      done < <(printf '%s\n' "${CI_PYTHON_TEST_PATHS}" | tr ',' '\n')
      ;;
    *)
      echo "Unknown CI_PYTHON_MODE: $mode (expected full, pantheon, or paths)" >&2
      return 1
      ;;
  esac
}
