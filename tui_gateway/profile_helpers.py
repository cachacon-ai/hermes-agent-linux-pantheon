"""Shared profile RPC helpers (imported inside handlers — see method_ctx.py)."""

from __future__ import annotations

from pathlib import Path


def atomic_write_profile_soul(profile_dir, content: str) -> None:
    """Replace SOUL.md atomically (same contract as REST PUT /api/profiles/.../soul)."""
    from utils import atomic_write_text

    soul_path = Path(profile_dir) / "SOUL.md"
    atomic_write_text(
        soul_path, content, preserve_mode=True, create_mode=0o644
    )


def profile_has_avatar(profile_dir) -> bool:
    assets = Path(profile_dir) / "assets"
    return any((assets / f"avatar.{ext}").is_file() for ext in ("png", "jpg", "webp"))


def read_profile_ui_meta(profile_dir) -> dict:
    try:
        import yaml as _yaml

        meta_path = Path(profile_dir) / "profile.yaml"
        if not meta_path.is_file():
            return {}
        with open(meta_path, "r", encoding="utf-8") as f:
            raw_meta = _yaml.safe_load(f) or {}
        ui_meta = raw_meta.get("ui_meta") if isinstance(raw_meta, dict) else None
        return ui_meta if isinstance(ui_meta, dict) else {}
    except Exception:
        return {}
