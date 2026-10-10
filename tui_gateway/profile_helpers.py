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


def resolve_profile_rpc(name: str):
    """Validate profile ``name`` and return ``(canon, profile_dir)``.

    Raises ``ValueError`` for invalid slugs and ``FileNotFoundError`` when
    the profile directory is missing.
    """
    from hermes_cli.profiles import (
        get_profile_dir,
        normalize_profile_name,
        validate_profile_name,
    )

    stripped = str(name or "").strip()
    if not stripped:
        raise ValueError("name required")
    canon = normalize_profile_name(stripped)
    if canon != "default":
        validate_profile_name(canon)
    profile_dir = get_profile_dir(canon)
    if not profile_dir.is_dir():
        raise FileNotFoundError(f"profile '{canon}' not found")
    return canon, profile_dir


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
