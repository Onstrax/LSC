"""Construcción del sobre JSON de respuesta y de errores uniformes.

Toda respuesta generada por la aplicación usa el mismo sobre, incluidos los
errores de validación. No se devuelven trazas, rutas privadas ni detalles de
configuración al usuario final.
"""

from __future__ import annotations

import uuid

from fastapi.responses import JSONResponse

from api.constants import (
    CAPTURE_PROTOCOL_VERSION,
    HTTP_SCHEMA_VERSION,
    STATUS_ACCEPTED,
    STATUS_ERROR,
    STATUS_RETRY,
)
from api.schemas.models import PredictResponse, TimingsMs


def new_request_id() -> str:
    return str(uuid.uuid4())


def build_envelope(
    *,
    status: str,
    mode: str,
    request_id: str | None = None,
    client_request_id: str | None = None,
    label_id: int | None = None,
    display_text: str | None = None,
    confidence: float | None = None,
    reason_codes: list[str] | None = None,
    model_version: str | None = None,
    feature_spec_hash: str | None = None,
    capture_protocol_version: str | None = None,
    timings_ms: dict | None = None,
) -> PredictResponse:
    """Crea un :class:`PredictResponse` validado."""
    return PredictResponse(
        schema_version=HTTP_SCHEMA_VERSION,
        request_id=request_id or new_request_id(),
        client_request_id=client_request_id,
        status=status,  # type: ignore[arg-type]
        label_id=label_id,
        display_text=display_text,
        confidence=confidence,
        reason_codes=list(reason_codes or []),
        model_version=model_version,
        feature_spec_hash=feature_spec_hash,
        capture_protocol_version=capture_protocol_version,
        mode=mode,  # type: ignore[arg-type]
        timings_ms=TimingsMs(**(timings_ms or {})),
    )


def error_response(
    *,
    http_status: int,
    mode: str,
    reason_codes: list[str],
    client_request_id: str | None = None,
    request_id: str | None = None,
    capture_protocol_version: str | None = None,
    headers: dict | None = None,
) -> JSONResponse:
    """Respuesta de error con el sobre uniforme y estado `error`."""
    envelope = build_envelope(
        status=STATUS_ERROR,
        mode=mode,
        request_id=request_id,
        client_request_id=client_request_id,
        reason_codes=reason_codes,
        capture_protocol_version=capture_protocol_version,
    )
    return JSONResponse(
        status_code=http_status,
        content=envelope.model_dump(),
        headers=headers or {},
    )


__all__ = [
    "new_request_id",
    "build_envelope",
    "error_response",
    "STATUS_ACCEPTED",
    "STATUS_RETRY",
    "STATUS_ERROR",
    "CAPTURE_PROTOCOL_VERSION",
]
