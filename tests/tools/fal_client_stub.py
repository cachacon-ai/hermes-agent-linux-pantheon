"""Test helpers for stubbing ``fal_client`` without production shortcuts."""

from __future__ import annotations

import sys
from typing import Any

import pytest


def monkeypatch_preserve_fal_client_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep a ``sys.modules['fal_client']`` test stub across ``lazy_deps.ensure``.

    ``ensure("image.fal")`` can import the real SDK and replace a stub module.
    Video/FAL tests install a stub with ``submit`` before calling production
    code; wrap ``ensure`` so that stub survives.
    """
    import tools.lazy_deps as lazy_deps

    real_ensure = lazy_deps.ensure

    def _ensure(feature: str, *, prompt: bool = True) -> Any:
        saved = sys.modules.get("fal_client")
        preserve = saved is not None and hasattr(saved, "submit")
        try:
            return real_ensure(feature, prompt=prompt)
        finally:
            if preserve and saved is not None:
                sys.modules["fal_client"] = saved

    monkeypatch.setattr(lazy_deps, "ensure", _ensure)
