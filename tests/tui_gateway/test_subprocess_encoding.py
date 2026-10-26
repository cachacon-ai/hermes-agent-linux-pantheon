"""Regression tests for UTF-8 encoding hardening in TUI subprocesses (#53137).

On Windows with a non-UTF-8 system locale (e.g. GBK on Chinese Windows),
text-mode subprocess reads defaulted to the locale encoding. When a child
process emitted bytes invalid in that locale, an unhandled UnicodeDecodeError
crashed the reader thread / gateway thread.

These tests assert that text-mode subprocess paths use UTF-8 with replacement.
PAN-78 moved quick-command exec to bounded_captured_run, so that test verifies
the helper's Popen kwargs instead of patching subprocess.run.

# Test pattern adapted from @devorun's PR #52700 (salvage convention).
"""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest

import tui_gateway.server as server


# ── helpers ──────────────────────────────────────────────────────────────

def _make_completed_process() -> MagicMock:
    """A CompletedProcess-like mock with str stdout/stderr (text=True contract)."""
    cp = MagicMock()
    cp.stdout = ""
    cp.stderr = ""
    cp.returncode = 0
    return cp


# ── _SlashWorker.Popen path ──────────────────────────────────────────────

def test_slash_worker_popen_uses_utf8_replace():
    """The slash-worker subprocess.Popen must pass encoding="utf-8" and
    errors="replace" so invalid bytes in child stdout/stderr don't raise
    UnicodeDecodeError inside the drain threads (#53137).
    """
    with patch.dict("sys.modules", {
        "hermes_constants": MagicMock(
            get_hermes_home=MagicMock(return_value="/tmp/hermes_test")
        ),
    }):
        with patch("subprocess.Popen") as mock_popen:
            mock_popen.return_value.stdout = MagicMock()
            mock_popen.return_value.stderr = MagicMock()

            from tui_gateway.server import _SlashWorker

            _SlashWorker(
                session_key="test_key",
                model="test-model",
            )

            assert mock_popen.called, "Popen was not invoked"
            kwargs = mock_popen.call_args[1]
            assert kwargs.get("encoding") == "utf-8", (
                f"slash-worker Popen must set encoding='utf-8' (got {kwargs.get('encoding')!r})"
            )
            assert kwargs.get("errors") == "replace", (
                f"slash-worker Popen must set errors='replace' (got {kwargs.get('errors')!r})"
            )


# ── cli.exec handler ─────────────────────────────────────────────────────

def test_cli_exec_uses_utf8_replace():
    """The cli.exec RPC handler runs `python -m hermes_cli.main` via
    subprocess.run; it must pass encoding="utf-8" and errors="replace"
    (#53137)."""
    handler = server._methods["cli.exec"]
    with patch("subprocess.run", return_value=_make_completed_process()) as mock_run:
        # Non-interactive argv that passes _cli_exec_blocked.
        resp = handler(1, {"argv": ["--version"]})
        assert mock_run.called, "subprocess.run was not invoked"
        kwargs = mock_run.call_args[1]
        assert kwargs.get("encoding") == "utf-8", (
            f"cli.exec subprocess.run must set encoding='utf-8' (got {kwargs.get('encoding')!r})"
        )
        assert kwargs.get("errors") == "replace", (
            f"cli.exec subprocess.run must set errors='replace' (got {kwargs.get('errors')!r})"
        )


# ── shell.exec handler ───────────────────────────────────────────────────

def test_shell_exec_uses_utf8_replace():
    """The shell.exec RPC handler runs an arbitrary shell command via
    subprocess.run; it must pass encoding="utf-8" and errors="replace"
    (#53137)."""
    handler = server._methods["shell.exec"]
    with patch("subprocess.run", return_value=_make_completed_process()) as mock_run:
        # A harmless, non-dangerous command that passes the approval gate.
        with patch("tools.approval.detect_hardline_command", return_value=(False, "")), \
             patch("tools.approval.detect_dangerous_command", return_value=(False, None, "")):
            resp = handler(1, {"command": "echo hello"})
        assert mock_run.called, "subprocess.run was not invoked"
        kwargs = mock_run.call_args[1]
        assert kwargs.get("encoding") == "utf-8", (
            f"shell.exec subprocess.run must set encoding='utf-8' (got {kwargs.get('encoding')!r})"
        )
        assert kwargs.get("errors") == "replace", (
            f"shell.exec subprocess.run must set errors='replace' (got {kwargs.get('errors')!r})"
        )


# ── quick-command exec path (via command.dispatch) ───────────────────────

def test_quick_command_exec_uses_utf8_replace():
    """command.dispatch uses bounded_captured_run with UTF-8 replacement decoding."""
    handler = server._methods["command.dispatch"]
    with patch("hermes_cli._subprocess_compat.subprocess.Popen") as mock_popen, \
         patch("tui_gateway.server._load_cfg", return_value={
             "quick_commands": {"runcmd": {"type": "exec", "command": "echo hi"}}
         }), \
         patch("tools.environments.local._sanitize_subprocess_env", return_value={"PATH": "/usr/bin"}):
        mock_popen.return_value.communicate.return_value = ("", "")
        mock_popen.return_value.returncode = 0
        resp = handler(1, {"name": "runcmd", "arg": "", "session_id": ""})
        assert mock_popen.called, "Popen was not invoked for quick-command exec"
        kwargs = mock_popen.call_args[1]
        assert kwargs.get("encoding") == "utf-8", (
            f"quick-command exec Popen must set encoding='utf-8' (got {kwargs.get('encoding')!r})"
        )
        assert kwargs.get("errors") == "replace", (
            f"quick-command exec Popen must set errors='replace' (got {kwargs.get('errors')!r})"
        )
