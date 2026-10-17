"""Codex ``codex_responses`` failed/cancelled soft failures must classify and rotate."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent.conversation_loop import _try_codex_soft_failure_pool_recovery
from agent.error_classifier import FailoverReason


def test_soft_failure_pool_recovery_classifies_and_rotates():
    agent = SimpleNamespace(
        api_mode="codex_responses",
        provider="openai-codex",
        model="gpt-5.3-codex",
    )
    recover_calls = []

    def _recover(**kwargs):
        recover_calls.append(kwargs)
        return True, False

    agent._recover_with_credential_pool = _recover
    agent._extract_api_error_context = lambda exc: {"message": str(exc)}
    agent._buffer_status = MagicMock()

    response = SimpleNamespace(
        status="failed",
        error={
            "code": "rate_limit_exceeded",
            "message": "Rate limit reached for requests in this organization",
        },
    )
    retry = SimpleNamespace(has_retried_429=False)

    assert _try_codex_soft_failure_pool_recovery(agent, response, retry) is True
    assert recover_calls
    assert recover_calls[0]["classified_reason"] == FailoverReason.rate_limit
    agent._buffer_status.assert_called_once()


def test_soft_failure_uses_only_supported_classify_kwargs():
    """Regression: unsupported kwargs to classify_api_error must not raise TypeError."""
    agent = SimpleNamespace(
        api_mode="codex_responses",
        provider="openai-codex",
        model="gpt-5.3-codex",
        base_url="https://chatgpt.com/backend-api/codex",
        api_key="should-not-be-passed-to-classifier",
    )
    agent._recover_with_credential_pool = lambda **kwargs: (False, kwargs["has_retried_429"])
    agent._extract_api_error_context = lambda exc: {}
    agent._buffer_status = MagicMock()

    response = SimpleNamespace(
        status="cancelled",
        error={"code": "usage_limit_reached", "message": "The usage limit has been reached"},
    )
    retry = SimpleNamespace(has_retried_429=False)

    # Must not raise TypeError from classify_api_error(**unsupported).
    assert _try_codex_soft_failure_pool_recovery(agent, response, retry) is False
