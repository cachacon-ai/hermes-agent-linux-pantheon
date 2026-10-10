"""Pinned memory per-turn injection (PAN-14 slice 4)."""

from __future__ import annotations

from pathlib import Path
import pytest

from agent.memory_manager import build_pinned_context_block
from agent.turn_context import compose_user_api_content
from tools.memory_tool import load_on_disk_store
from tools.profile_context_store import write_document_entries


@pytest.fixture
def profile_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    (home / "memories").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    return home


def test_pins_injected_when_prefetch_empty(profile_home):
    write_document_entries("pins", ["pin fact A"])
    from tools.profile_context_store import entries_for_turn_injection

    block = build_pinned_context_block(entries_for_turn_injection())
    out = compose_user_api_content("hello", "", "", block)
    assert out is not None
    assert "<pinned-memory>" in out
    assert "pin fact A" in out
    assert "<memory-context>" not in out


def test_pins_absent_from_cached_system_prompt_snapshot(profile_home):
    write_document_entries("pins", ["pin-only content"])
    write_document_entries("memory", ["memory note"])
    store = load_on_disk_store()
    snap = store._system_prompt_snapshot
    assert "pin-only content" not in (snap.get("memory") or "")
    assert "pin-only content" not in (snap.get("user") or "")
    assert "memory note" in (snap.get("memory") or "")


def test_pins_survive_compression_style_user_recompose(profile_home):
    """Compression drops old user api_content; next turn re-injects pins."""
    write_document_entries("pins", ["survives compression"])
    from tools.profile_context_store import entries_for_turn_injection

    block = build_pinned_context_block(entries_for_turn_injection())
    first = compose_user_api_content("turn one", "prefetch", "", block)
    assert "survives compression" in first
    # Simulate compressed history: user message content rewritten, sidecar dropped.
    second = compose_user_api_content("summary turn", "", "", block)
    assert "survives compression" in second

