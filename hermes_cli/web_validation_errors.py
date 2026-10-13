"""Safe FastAPI request-validation error responses for the dashboard server."""

from __future__ import annotations

from typing import Any

from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.requests import Request

_BINARY_INPUT_PLACEHOLDER = "<binary request body omitted>"


def _sanitize_validation_value(value: Any) -> Any:
    if isinstance(value, bytes):
        return _BINARY_INPUT_PLACEHOLDER
    if isinstance(value, bytearray):
        return _BINARY_INPUT_PLACEHOLDER
    if isinstance(value, dict):
        return {key: _sanitize_validation_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_validation_value(item) for item in value]
    return value


def sanitize_validation_errors(errors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return a JSON-serializable copy of Pydantic/FastAPI validation errors."""
    return [_sanitize_validation_value(dict(error)) for error in errors]


def _transcribe_multipart_hint(request: Request, errors: list[dict[str, Any]]) -> None:
    if request.url.path != "/api/audio/transcribe":
        return
    content_type = (request.headers.get("content-type") or "").lower()
    if "multipart/form-data" not in content_type:
        return
    for error in errors:
        if tuple(error.get("loc") or ()) == ("body",):
            error["msg"] = (
                "Expected application/json with 'data_url' (base64 data URL) "
                "and optional 'mime_type', not multipart/form-data."
            )
            break


def request_validation_error_response(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Build a 422 response without decoding binary validation inputs as UTF-8."""
    detail = sanitize_validation_errors(exc.errors())
    _transcribe_multipart_hint(request, detail)
    return JSONResponse(
        status_code=422,
        content={"detail": jsonable_encoder(detail)},
    )


async def handle_request_validation_error(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    return request_validation_error_response(request, exc)
