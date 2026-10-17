"""Shared helpers for dashboard auth routes and middleware."""

from fastapi import Request


def client_ip(request: Request) -> str:
    """ASGI peer address for rate limits and auth audit.

    Never parse client-supplied ``X-Forwarded-For`` here: direct clients can
    spoof it. Trusted proxy normalization belongs upstream.
    """
    return request.client.host if request.client else ""
