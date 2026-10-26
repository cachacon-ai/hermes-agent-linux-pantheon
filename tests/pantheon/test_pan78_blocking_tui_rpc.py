"""PAN-78: blocking TUI RPC work must not run on the stdio/WS reader thread."""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time

import pytest

from hermes_cli._subprocess_compat import bounded_captured_run
from tui_gateway import server


@pytest.mark.parametrize(
    "blocking_method",
    ["pdf.attach", "command.dispatch"],
)
def test_pdf_attach_and_command_dispatch_are_pool_routed(blocking_method):
    assert blocking_method in server._LONG_HANDLERS


def test_pdf_attach_subprocess_timeout_below_client_rpc_deadline():
    """Server pdftoppm ceiling must leave headroom under the client RPC timer.

    Clients (ui-tui gatewayClient, apps/shared JsonRpcGatewayClient) start a
    120s deadline before dispatch; the subprocess timer starts only after pool
    scheduling + validation. Equal ceilings make a real 5028 unreachable.
    """
    assert (
        server._PDF_ATTACH_SUBPROCESS_TIMEOUT_S
        < server._PDF_ATTACH_CLIENT_RPC_DEADLINE_S
    )
    # Comfortable margin for pool queue + validation + response write.
    assert (
        server._PDF_ATTACH_CLIENT_RPC_DEADLINE_S
        - server._PDF_ATTACH_SUBPROCESS_TIMEOUT_S
        >= 15
    )


def _building_session(tmp_path, sid: str) -> dict:
    session = {
        "agent": None,
        "agent_ready": threading.Event(),
        "agent_error": None,
        "attached_images": [],
        "cwd": str(tmp_path),
        "history": [],
        "history_lock": threading.RLock(),
        "history_version": 0,
        "image_counter": 0,
        "profile_home": str(tmp_path),
        "running": False,
        "session_key": sid,
        "transport": None,
    }
    server._sessions[sid] = session
    return session


def test_pdf_attach_timeout_delivers_5028_before_client_deadline(
    monkeypatch, tmp_path
):
    """Advance both budgets: server fires first; client still receives 5028."""
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    # Stall past the patched server timeout but well under the client deadline.
    if sys.platform == "win32":
        pdftoppm = fake_bin / "pdftoppm.bat"
        pdftoppm.write_text(
            f"@echo off\r\n{sys.executable} -c \"import time; time.sleep(30)\"\r\n",
            encoding="utf-8",
        )
    else:
        pdftoppm = fake_bin / "pdftoppm"
        pdftoppm.write_text(
            f"#!{sys.executable}\nimport time\ntime.sleep(30)\n",
            encoding="utf-8",
        )
        pdftoppm.chmod(0o755)
    monkeypatch.setenv("PATH", f"{fake_bin}{os.pathsep}{os.environ.get('PATH', '')}")
    monkeypatch.setattr(server, "_start_agent_build", lambda sid, session: None)
    monkeypatch.setattr(server, "_PDF_ATTACH_SUBPROCESS_TIMEOUT_S", 0.4)

    sid = "pan78-pdf-timeout"
    _building_session(tmp_path, sid)
    client_deadline_s = 3.0  # stands in for the real 120s client RPC timer
    pdf_b64 = "JVBERi0xLjQK"  # %PDF-1.4

    start = time.monotonic()
    try:
        resp = server.handle_request(
            {
                "id": "1",
                "method": "pdf.attach",
                "params": {
                    "session_id": sid,
                    "content_base64": pdf_b64,
                    "filename": "slow.pdf",
                },
            }
        )
    finally:
        server._sessions.pop(sid, None)
    elapsed = time.monotonic() - start

    assert "error" in resp, resp
    assert resp["error"]["code"] == 5028
    assert "timed out" in resp["error"]["message"].lower()
    assert elapsed < client_deadline_s, (
        f"5028 took {elapsed:.2f}s — client deadline {client_deadline_s}s "
        "would have already rejected the call"
    )


def test_command_dispatch_exec_timeout_returns_error(monkeypatch):
    """Response-shape unit test: TimeoutExpired maps to error 5016."""
    monkeypatch.setattr(
        server,
        "_load_cfg",
        lambda: {"quick_commands": {"slow": {"type": "exec", "command": "sleep 99"}}},
    )

    def _timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=kwargs.get("args") or args[:1], timeout=30)

    monkeypatch.setattr(
        "hermes_cli._subprocess_compat.bounded_captured_run", _timeout
    )

    resp = server.handle_request(
        {"id": "1", "method": "command.dispatch", "params": {"name": "slow"}}
    )

    assert resp["error"]["code"] == 5016
    assert "timed out" in resp["error"]["message"].lower()


def test_command_dispatch_timeout_bounded_with_pipe_holding_descendant(
    monkeypatch, tmp_path
):
    """Real descendant-holds-pipes case must still return 5016 promptly.

    On Windows, ``subprocess.run(..., capture_output=True, timeout=N)`` can
    hang forever after killing the direct child when a descendant still holds
    duplicated stdout/stderr handles. Quick commands use shell=True, so this
    is the failure mode that could pin every shared RPC worker.
    """
    script = tmp_path / "pipe_hold.py"
    script.write_text(
        "import subprocess, sys, time\n"
        "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        "time.sleep(60)\n",
        encoding="utf-8",
    )
    cmd = f"{sys.executable} {script}"
    monkeypatch.setattr(
        server,
        "_load_cfg",
        lambda: {"quick_commands": {"sticky": {"type": "exec", "command": cmd}}},
    )
    monkeypatch.setattr(server, "_QUICK_COMMAND_EXEC_TIMEOUT_S", 0.5)

    start = time.monotonic()
    resp = server.handle_request(
        {"id": "1", "method": "command.dispatch", "params": {"name": "sticky"}}
    )
    elapsed = time.monotonic() - start

    assert resp["error"]["code"] == 5016
    assert "timed out" in resp["error"]["message"].lower()
    # timeout + tree-kill + 1s drain + slack. Pre-fix hang is unbounded.
    assert elapsed < 15, f"quick-command timeout hung for {elapsed:.1f}s"


@pytest.mark.windows_only
def test_bounded_captured_run_windows_descendant_pipe_hold_returns():
    """Windows-lane proof: tree-kill + bounded drain beats pipe-hold hang."""
    script = (
        "import subprocess, sys, time\n"
        "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        "time.sleep(60)\n"
    )
    start = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired):
        bounded_captured_run(
            [sys.executable, "-c", script],
            timeout=0.5,
        )
    elapsed = time.monotonic() - start
    assert elapsed < 15, f"bounded_captured_run hung for {elapsed:.1f}s on Windows"
