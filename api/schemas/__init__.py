"""Esquemas del contrato HTTP `lsc.http.v1` (propuesta de Juan).

La versión acordada se congela en `api/schemas/openapi.json` y Ortiz la
incorpora en `contracts/`. No se mantienen dos contratos activos distintos.
"""

from api.schemas.models import (
    ConfigResponse,
    HealthResponse,
    PredictMetadata,
    PredictResponse,
    TimingsMs,
)

__all__ = [
    "PredictMetadata",
    "PredictResponse",
    "HealthResponse",
    "ConfigResponse",
    "TimingsMs",
]
