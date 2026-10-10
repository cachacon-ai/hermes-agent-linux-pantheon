"""Profile-scoped memory/user/pins documents for Pantheon context RPCs.

Concurrency: all writes acquire :meth:`MemoryStore._file_lock` on the target
file (same lock the built-in ``memory`` tool uses). Last write wins unless the
caller supplies ``expected_revision`` (content hash compare-and-swap).

Edits apply on disk immediately; the live agent's in-memory ``MemoryStore`` and
frozen system-prompt snapshot refresh on the next new session (by design).
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from hermes_constants import get_hermes_home
from tools.memory_tool import (
    ENTRY_DELIMITER,
    MemoryStore,
    _scan_memory_content,
    get_builtin_memory_config,
    get_memory_dir,
)

logger = logging.getLogger(__name__)

PINNED_FILENAME = "PINNED.md"

DEFAULT_PINS_CHAR_LIMIT = 2000
DEFAULT_PINS_MAX_COUNT = 20

LIMIT_MIN_CHARS = 200
LIMIT_MAX_CHARS = 20000
PINS_MAX_COUNT_CEILING = 100

TARGET_MEMORY = "memory"
TARGET_USER = "user"
TARGET_PINS = "pins"

_VALID_TARGETS = frozenset({TARGET_MEMORY, TARGET_USER, TARGET_PINS})

_PINNED_BLOCK_OPEN = "<pinned-memory>"
_PINNED_BLOCK_CLOSE = "</pinned-memory>"

_FORBIDDEN_ENTRY_SUBSTRINGS = (
    ENTRY_DELIMITER,
    _PINNED_BLOCK_OPEN,
    _PINNED_BLOCK_CLOSE,
)


def document_path(target: str) -> Path:
    mem_dir = get_memory_dir()
    if target == TARGET_USER:
        return mem_dir / "USER.md"
    if target == TARGET_PINS:
        return mem_dir / PINNED_FILENAME
    return mem_dir / "MEMORY.md"


def entries_revision(entries: List[str]) -> str:
    """Stable content hash for compare-and-swap (full serialized document)."""
    payload = ENTRY_DELIMITER.join(entries) if entries else ""
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_document_entries(target: str) -> Tuple[List[str], bool]:
    path = document_path(target)
    return MemoryStore._read_entries_checked(path)


def _normalize_entries(raw: List[str]) -> Tuple[List[str], Optional[str]]:
    cleaned = [e.strip() for e in raw if isinstance(e, str) and e.strip()]
    for entry in cleaned:
        for forbidden in _FORBIDDEN_ENTRY_SUBSTRINGS:
            if forbidden in entry:
                return [], (
                    f"Entry must not contain the delimiter or pinned-memory fence "
                    f"({forbidden!r})."
                )
    return list(dict.fromkeys(cleaned)), None


def _char_count(entries: List[str]) -> int:
    if not entries:
        return 0
    return len(ENTRY_DELIMITER.join(entries))


def get_context_limits(config: Optional[Dict[str, Any]] = None) -> Dict[str, int]:
    mem_cfg = get_builtin_memory_config(config)
    return {
        "memory": int(mem_cfg.get("memory_char_limit", 2200)),
        "user": int(mem_cfg.get("user_char_limit", 1375)),
        "pins": int(mem_cfg.get("pins_char_limit", DEFAULT_PINS_CHAR_LIMIT)),
        "pins_max_count": int(mem_cfg.get("pins_max_count", DEFAULT_PINS_MAX_COUNT)),
    }


def usage_payload(
    target: str, entries: List[str], limits: Dict[str, int]
) -> Dict[str, Any]:
    chars = _char_count(entries)
    if target == TARGET_PINS:
        limit = limits["pins"]
        return {
            "chars": chars,
            "limit": limit,
            "count": len(entries),
            "max_count": limits["pins_max_count"],
        }
    limit = limits["user"] if target == TARGET_USER else limits["memory"]
    return {"chars": chars, "limit": limit}


def _validate_write_entries(entries: List[str]) -> Optional[str]:
    for entry in entries:
        err = _scan_memory_content(entry)
        if err:
            return err
    return None


def _check_capacity(
    target: str, entries: List[str], limits: Dict[str, int]
) -> Optional[str]:
    usage = usage_payload(target, entries, limits)
    if target == TARGET_PINS:
        if len(entries) > limits["pins_max_count"]:
            return (
                f"Pin count {len(entries)} exceeds max {limits['pins_max_count']}."
            )
        if usage["chars"] > limits["pins"]:
            return (
                f"Pinned memory size {usage['chars']} chars exceeds limit "
                f"{limits['pins']}."
            )
        return None
    if usage["chars"] > usage["limit"]:
        label = "USER" if target == TARGET_USER else "MEMORY"
        return (
            f"{label} size {usage['chars']} chars exceeds limit {usage['limit']}."
        )
    return None


def write_document_entries(
    target: str,
    entries: List[str],
    *,
    expected_revision: Optional[str] = None,
    limits: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """Persist entries under file lock. Returns applied/usage/revision dict."""
    if target not in _VALID_TARGETS:
        return {"applied": False, "error": f"Invalid target {target!r}."}

    limits = limits or get_context_limits()
    normalized, norm_err = _normalize_entries(entries)
    if norm_err:
        return {
            "applied": False,
            "error": norm_err,
            "usage": usage_payload(target, normalized, limits),
        }

    scan_err = _validate_write_entries(normalized)
    if scan_err:
        return {
            "applied": False,
            "error": scan_err,
            "usage": usage_payload(target, normalized, limits),
        }

    cap_err = _check_capacity(target, normalized, limits)
    if cap_err:
        return {
            "applied": False,
            "error": cap_err,
            "usage": usage_payload(target, normalized, limits),
        }

    path = document_path(target)
    path.parent.mkdir(parents=True, exist_ok=True)

    with MemoryStore._file_lock(path):
        current, read_ok = read_document_entries(target)
        if not read_ok:
            return {
                "applied": False,
                "error": f"Could not read {path.name}; try again.",
                "usage": usage_payload(target, current, limits),
            }
        current_rev = entries_revision(current)
        if expected_revision is not None and expected_revision != current_rev:
            return {
                "applied": False,
                "error": "revision_conflict",
                "revision": current_rev,
                "usage": usage_payload(target, current, limits),
            }

        MemoryStore._write_file(path, normalized)

    new_rev = entries_revision(normalized)
    return {
        "applied": True,
        "usage": usage_payload(target, normalized, limits),
        "entry_count": len(normalized),
        "revision": new_rev,
    }


def read_context_snapshot(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Load memory/user/pins entries + revisions for profiles.context.get."""
    limits = get_context_limits(config)

    def _block(target: str) -> Dict[str, Any]:
        entries, ok = read_document_entries(target)
        if not ok:
            entries = []
        return {
            "entries": entries,
            "revision": entries_revision(entries),
            "usage": usage_payload(target, entries, limits),
        }

    return {
        "memory": _block(TARGET_MEMORY),
        "user": _block(TARGET_USER),
        "pins": _block(TARGET_PINS),
        "limits": limits,
    }


def clamp_limit_value(value: Any, *, field: str) -> Tuple[Optional[int], Optional[str]]:
    try:
        n = int(value)
    except (TypeError, ValueError):
        return None, f"{field} must be an integer."
    if field == "pins_max_count":
        if n < 1 or n > PINS_MAX_COUNT_CEILING:
            return None, f"{field} must be between 1 and {PINS_MAX_COUNT_CEILING}."
        return n, None
    if n < LIMIT_MIN_CHARS or n > LIMIT_MAX_CHARS:
        return None, f"{field} must be between {LIMIT_MIN_CHARS} and {LIMIT_MAX_CHARS}."
    return n, None


def update_context_limits(
    updates: Dict[str, Any],
    *,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Apply limit changes after verifying current usage fits."""
    current = get_context_limits(config)
    merged = dict(current)
    errors: List[str] = []

    for key in ("memory", "user", "pins", "pins_max_count"):
        if key not in updates or updates[key] is None:
            continue
        val, err = clamp_limit_value(updates[key], field=key)
        if err:
            errors.append(err)
        elif val is not None:
            merged[key] = val

    if errors:
        return {"applied": False, "error": "; ".join(errors), "limits": current}

    snap = read_context_snapshot(config)
    if merged["memory"] < snap["memory"]["usage"]["chars"]:
        return {
            "applied": False,
            "error": (
                f"memory limit {merged['memory']} is below current usage "
                f"{snap['memory']['usage']['chars']} chars."
            ),
            "limits": current,
        }
    if merged["user"] < snap["user"]["usage"]["chars"]:
        return {
            "applied": False,
            "error": (
                f"user limit {merged['user']} is below current usage "
                f"{snap['user']['usage']['chars']} chars."
            ),
            "limits": current,
        }
    if merged["pins"] < snap["pins"]["usage"]["chars"]:
        return {
            "applied": False,
            "error": (
                f"pins limit {merged['pins']} is below current usage "
                f"{snap['pins']['usage']['chars']} chars."
            ),
            "limits": current,
        }
    if merged["pins_max_count"] < snap["pins"]["usage"]["count"]:
        return {
            "applied": False,
            "error": (
                f"pins_max_count {merged['pins_max_count']} is below current "
                f"pin count {snap['pins']['usage']['count']}."
            ),
            "limits": current,
        }

    return {"applied": True, "limits": merged}


def persist_context_limits_to_config(
    new_limits: Dict[str, int],
    *,
    previous: Optional[Dict[str, int]] = None,
) -> None:
    """Write changed limit keys to config.yaml in one round-trip save."""
    from hermes_cli.config import (
        _CONFIG_LOCK,
        get_config_path,
        read_raw_config,
        require_readable_config_before_write,
    )

    previous = previous or get_context_limits(read_raw_config())
    key_paths = {
        "memory": "memory.memory_char_limit",
        "user": "memory.user_char_limit",
        "pins": "memory.pins_char_limit",
        "pins_max_count": "memory.pins_max_count",
    }
    pending = {
        path: new_limits[key]
        for key, path in key_paths.items()
        if int(new_limits[key]) != int(previous.get(key, new_limits[key]))
    }
    if not pending:
        return

    config_path = get_config_path()
    with _CONFIG_LOCK:
        require_readable_config_before_write(config_path)
        _atomic_roundtrip_yaml_apply_updates(config_path, pending)


def _atomic_roundtrip_yaml_apply_updates(
    path: Path, updates: Dict[str, Any]
) -> None:
    """Apply several dotted-key updates in a single ruamel round-trip write."""
    import os
    import tempfile

    from ruamel.yaml import YAML
    from ruamel.yaml.comments import CommentedMap

    from hermes_cli.config import _greedy_literal_match, _split_key_path
    from utils import (
        _preserve_file_mode,
        _preserve_file_owner,
        _restore_file_mode,
        _restore_file_owner,
        atomic_replace,
    )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    yaml_rt = YAML(typ="rt")
    yaml_rt.preserve_quotes = True
    yaml_rt.allow_unicode = True
    yaml_rt.default_flow_style = False
    yaml_rt.indent(mapping=2, sequence=4, offset=2)

    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            config = yaml_rt.load(f) or CommentedMap()
    else:
        config = CommentedMap()

    if not isinstance(config, CommentedMap):
        config = CommentedMap(config)

    for key_path, value in updates.items():
        current = config
        keys = _split_key_path(key_path)
        i = 0
        while True:
            remaining = keys[i:]
            seg, consumed = remaining[0], 1
            match = _greedy_literal_match(dict(current), remaining)
            if match is not None:
                seg, consumed = match
            if i + consumed == len(keys):
                current[seg] = value
                break
            next_value = current.get(seg)
            if not isinstance(next_value, CommentedMap):
                next_value = CommentedMap()
                current[seg] = next_value
            current = next_value
            i += consumed

    original_mode = _preserve_file_mode(path)
    original_owner = _preserve_file_owner(path)
    fd, tmp_path = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=f".{path.stem}_",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            yaml_rt.dump(config, f)
            f.flush()
            os.fsync(f.fileno())
        real_path = atomic_replace(tmp_path, path)
        real_path_obj = Path(real_path)
        _restore_file_owner(real_path_obj, original_owner)
        _restore_file_mode(real_path_obj, original_mode)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def entries_for_turn_injection(
    config: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Pinned entries to inject this turn, capped by configured limits."""
    limits = get_context_limits(config)
    entries, ok = read_document_entries(TARGET_PINS)
    if not ok or not entries:
        return []
    max_count = limits["pins_max_count"]
    char_limit = limits["pins"]
    selected: List[str] = []
    used = 0
    for entry in entries:
        if _scan_memory_content(entry):
            continue
        if len(selected) >= max_count:
            break
        extra = len(entry) if not selected else len(ENTRY_DELIMITER) + len(entry)
        if used + extra > char_limit:
            break
        selected.append(entry)
        used += extra
    return selected
