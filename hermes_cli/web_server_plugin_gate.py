"""Runtime plugin consent gate shared by HTTP, WebSocket upgrades, and static assets."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Optional

from starlette import status
from starlette._utils import get_route_path

from hermes_cli.config import cfg_get, load_config_readonly
from hermes_constants import (
    get_hermes_home,
    get_process_hermes_home,
    hermes_home_key,
    reset_hermes_home_override,
    set_hermes_home_override,
)


def _plugin_api_mount_skip_reason(
    plugin: dict[str, Any], enabled_set: set, disabled_set: set
) -> Optional[str]:
    """Why a plugin backend must not run, or None when it may."""
    source, plugin_name = plugin.get("source"), plugin.get("name", "")
    if source in ("user", "bundled") and plugin_name in disabled_set:
        return "explicitly disabled"
    if source == "user" and plugin_name not in enabled_set:
        return "not in plugins.enabled"
    return None


@contextmanager
def _launch_profile_scope():
    """Resolve plugin enable lists as the launch profile, not a scoped request."""
    token = set_hermes_home_override(None)
    try:
        yield
    finally:
        reset_hermes_home_override(token)


def _plugin_enable_sets_here() -> tuple[set, set]:
    try:
        config = load_config_readonly()
    except Exception:
        config = {}
    enabled = cfg_get(config, "plugins", "enabled", default=[])
    disabled = cfg_get(config, "plugins", "disabled", default=[])
    return (
        set(enabled) if isinstance(enabled, list) else set(),
        set(disabled) if isinstance(disabled, list) else set(),
    )


def _plugin_consent_withheld(plugin: dict[str, Any]) -> bool:
    """True when the active scope and launch profile both disallow serving ``plugin``."""
    if _plugin_api_mount_skip_reason(plugin, *_plugin_enable_sets_here()):
        return True
    if hermes_home_key(get_hermes_home()) == hermes_home_key(get_process_hermes_home()):
        return False
    with _launch_profile_scope():
        return _plugin_api_mount_skip_reason(plugin, *_plugin_enable_sets_here()) is not None


def _plugin_api_name(scope) -> str:
    path = get_route_path(scope)
    parts = path.split("/")
    return parts[3] if path.startswith("/api/plugins/") and len(parts) >= 4 else ""


def _plugin_api_request_refused(name: str) -> bool:
    from hermes_cli.web_server import _get_dashboard_plugins

    plugin = next((p for p in _get_dashboard_plugins() if p.get("name") == name), None)
    return _plugin_consent_withheld(plugin or {"name": name, "source": "user"})


class PluginApiWebSocketGate:
    """Apply :func:`_plugin_api_request_refused` to WebSocket upgrades (#134889 PLG-8)."""

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] == "websocket":
            name = _plugin_api_name(scope)
            if name and _plugin_api_request_refused(name):
                await send(
                    {
                        "type": "websocket.close",
                        "code": status.WS_1008_POLICY_VIOLATION,
                    }
                )
                return
        await self.app(scope, receive, send)
