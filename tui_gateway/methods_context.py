"""Profile context JSON-RPC handlers (PAN-14 slices 3–4).

Handlers delegate to ``tools.profile_context_store`` and reuse the built-in
memory file lock for concurrency with the ``memory`` tool.
"""

from __future__ import annotations

from .method_ctx import HandlerRegistry

_registry = HandlerRegistry()
method = _registry.method


@method("profiles.context.get")
def _(rid, params: dict) -> dict:
    """Return soul + memory/user/pins documents for a profile editor."""
    try:
        from tui_gateway.profile_helpers import resolve_profile_rpc

        _canon, profile_dir = resolve_profile_rpc(params.get("name"))
    except ValueError as exc:
        return _err(rid, 4063 if "required" in str(exc) else 4065, str(exc))
    except FileNotFoundError as exc:
        return _err(rid, 4064, str(exc))
    except Exception as exc:
        return _err(rid, 5063, str(exc))

    try:
        from hermes_constants import reset_hermes_home_override, set_hermes_home_override
        from hermes_cli.config import load_config
        from tools.profile_context_store import read_context_snapshot

        token = set_hermes_home_override(str(profile_dir))
        try:
            cfg = load_config() or {}
            snap = read_context_snapshot(cfg)
            soul_path = profile_dir / "SOUL.md"
            soul = ""
            if soul_path.is_file():
                soul = soul_path.read_text(encoding="utf-8", errors="replace")
            return _ok(
                rid,
                {
                    "soul": soul,
                    "memory": snap["memory"],
                    "user": snap["user"],
                    "pins": snap["pins"],
                    "limits": snap["limits"],
                },
            )
        finally:
            reset_hermes_home_override(token)
    except Exception as exc:
        return _err(rid, 5063, str(exc))


@method("profiles.context.set")
def _(rid, params: dict) -> dict:
    """Replace memory, user, or pins entries (optional revision CAS)."""
    try:
        from tui_gateway.profile_helpers import resolve_profile_rpc

        _canon, profile_dir = resolve_profile_rpc(params.get("name"))
    except ValueError as exc:
        return _err(rid, 4063 if "required" in str(exc) else 4065, str(exc))
    except FileNotFoundError as exc:
        return _err(rid, 4064, str(exc))
    except Exception as exc:
        return _err(rid, 5063, str(exc))

    target = str(params.get("target") or "").strip().lower()
    if target not in ("memory", "user", "pins"):
        return _err(rid, 4066, "target must be memory, user, or pins")
    raw_entries = params.get("entries")
    if not isinstance(raw_entries, list) or not all(
        isinstance(e, str) for e in raw_entries
    ):
        return _err(rid, 4067, "entries must be a list of strings")
    expected_revision = params.get("expected_revision")
    if expected_revision is not None and not isinstance(expected_revision, str):
        return _err(rid, 4068, "expected_revision must be a string when provided")

    try:
        from hermes_constants import reset_hermes_home_override, set_hermes_home_override
        from hermes_cli.config import load_config
        from tools.profile_context_store import (
            get_context_limits,
            write_document_entries,
        )

        token = set_hermes_home_override(str(profile_dir))
        try:
            cfg = load_config() or {}
            limits = get_context_limits(cfg)
            result = write_document_entries(
                target,
                raw_entries,
                expected_revision=expected_revision,
                limits=limits,
            )
            return _ok(rid, result)
        finally:
            reset_hermes_home_override(token)
    except Exception as exc:
        return _err(rid, 5063, str(exc))


@method("profiles.context.limits")
def _(rid, params: dict) -> dict:
    """Update memory/user/pins size limits in the profile config.yaml."""
    try:
        from tui_gateway.profile_helpers import resolve_profile_rpc

        _canon, profile_dir = resolve_profile_rpc(params.get("name"))
    except ValueError as exc:
        return _err(rid, 4063 if "required" in str(exc) else 4065, str(exc))
    except FileNotFoundError as exc:
        return _err(rid, 4064, str(exc))
    except Exception as exc:
        return _err(rid, 5063, str(exc))

    updates = {
        k: params.get(k)
        for k in ("memory", "user", "pins", "pins_max_count")
        if k in params
    }
    if not updates:
        return _err(rid, 4069, "at least one limit field required")

    try:
        from hermes_constants import reset_hermes_home_override, set_hermes_home_override
        from hermes_cli.config import load_config
        from tools.profile_context_store import (
            get_context_limits,
            persist_context_limits_to_config,
            update_context_limits,
        )

        token = set_hermes_home_override(str(profile_dir))
        try:
            cfg = load_config() or {}
            limits_before = get_context_limits(cfg)
            outcome = update_context_limits(updates, config=cfg)
            if not outcome.get("applied"):
                return _ok(rid, outcome)
            new_limits = outcome["limits"]
            persist_context_limits_to_config(
                new_limits, previous=limits_before
            )
            return _ok(rid, {"applied": True, "limits": new_limits})
        finally:
            reset_hermes_home_override(token)
    except Exception as exc:
        return _err(rid, 5063, str(exc))


def register(server) -> None:
    _registry.install(server)
