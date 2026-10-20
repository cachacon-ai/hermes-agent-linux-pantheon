"""MiniMax M3.1 Flash: catalog, validation and adaptive-thinking wire format (PAN-70)."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from agent.anthropic_adapter import build_anthropic_kwargs
from hermes_cli.models import _PROVIDER_MODELS, validate_requested_model

MESSAGES = [{"role": "user", "content": "hello"}]


def _kwargs(model: str, reasoning_config: dict | None):
    return build_anthropic_kwargs(
        model=model,
        messages=MESSAGES,
        tools=None,
        max_tokens=4096,
        reasoning_config=reasoning_config,
    )


@pytest.mark.parametrize("provider", ["minimax", "minimax-oauth"])
def test_catalog_lists_m31_flash(provider: str) -> None:
    assert "MiniMax-M3.1-Flash" in _PROVIDER_MODELS[provider]


@pytest.mark.parametrize("provider", ["minimax", "minimax-oauth"])
def test_m31_flash_validates_against_static_catalog(provider: str) -> None:
    with patch("hermes_cli.models.fetch_api_models", return_value=None):
        result = validate_requested_model("MiniMax-M3.1-Flash", provider)
    assert result["accepted"] is True
    assert result["recognized"] is True


@pytest.mark.parametrize("model", ["MiniMax-M3.1-Flash", "minimax/MiniMax-M3.1-Flash"])
def test_m31_flash_uses_adaptive_thinking(model: str) -> None:
    kwargs = _kwargs(model, {"enabled": True, "effort": "high"})
    assert kwargs["thinking"] == {"type": "adaptive"}
    assert kwargs["output_config"]["effort"] == "high"


def test_m31_flash_reasoning_off_maps_to_low_effort() -> None:
    kwargs = _kwargs("MiniMax-M3.1-Flash", {"enabled": False})
    assert kwargs["thinking"] == {"type": "adaptive"}
    assert kwargs["output_config"]["effort"] == "low"


def test_older_minimax_keeps_previous_thinking_shape() -> None:
    kwargs = _kwargs("MiniMax-M3", {"enabled": True, "effort": "high"})
    assert kwargs.get("thinking", {}).get("type") != "adaptive"
