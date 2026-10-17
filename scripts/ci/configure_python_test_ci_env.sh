#!/usr/bin/env bash
# Write CI_PYTHON_* variables to GITHUB_ENV for test workflows.
set -euo pipefail

if [[ -z "${GITHUB_ENV:-}" ]]; then
  echo "GITHUB_ENV is not set" >&2
  exit 1
fi

if [[ -n "${INPUT_PYTHON_MODE:-}" ]]; then
  echo "CI_PYTHON_MODE=${INPUT_PYTHON_MODE}" >> "$GITHUB_ENV"
elif [[ "${GITHUB_EVENT_NAME:-}" == "schedule" ]]; then
  echo "CI_PYTHON_MODE=full" >> "$GITHUB_ENV"
elif [[ "${GITHUB_EVENT_NAME:-}" == "workflow_dispatch" ]]; then
  echo "CI_PYTHON_MODE=${WORKFLOW_DISPATCH_PYTHON_MODE:-full}" >> "$GITHUB_ENV"
elif [[ -n "${REPO_CI_PYTHON_MODE:-}" ]]; then
  echo "CI_PYTHON_MODE=${REPO_CI_PYTHON_MODE}" >> "$GITHUB_ENV"
fi

if [[ -n "${REPO_CI_PYTHON_TEST_PATHS:-}" ]]; then
  echo "CI_PYTHON_TEST_PATHS=${REPO_CI_PYTHON_TEST_PATHS}" >> "$GITHUB_ENV"
fi

if [[ -n "${REPO_CI_PYTHON_EXTRA_ARGS:-}" ]]; then
  echo "CI_PYTHON_EXTRA_ARGS=${REPO_CI_PYTHON_EXTRA_ARGS}" >> "$GITHUB_ENV"
fi
