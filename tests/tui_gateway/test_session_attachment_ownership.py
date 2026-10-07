"""Shared profile staging paths must stay individually owned by sessions."""

from tui_gateway import server


def test_file_attach_retries_a_path_taken_after_the_unique_name_check(monkeypatch, tmp_path):
    home = tmp_path / "profile"
    staged_dir = home / "attachments"
    staged_dir.mkdir(parents=True)
    occupied = staged_dir / "report.txt"
    occupied.write_text("another session's upload")
    retry = staged_dir / "report-2.txt"
    candidates = iter((occupied, retry))
    monkeypatch.setattr(server, "_unique_attachment_path", lambda *_: next(candidates))

    path, uploaded = server._stage_session_file_attachment(
        {"cwd": str(tmp_path / "workspace"), "profile_home": str(home)},
        raw_path="",
        data_url="data:text/plain;base64,bmV3IHVwbG9hZA==",
        name="report.txt",
    )

    assert uploaded is True
    assert path == retry
    assert occupied.read_text() == "another session's upload"
    assert retry.read_text() == "new upload"
