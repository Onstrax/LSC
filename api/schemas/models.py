"""Modelos Pydantic del contrato HTTP `lsc.http.v1`.

Reglas (SDD §9, §10, §12):
- Toda respuesta usa el mismo sobre JSON, incluidos errores de validación.
- Los campos anulables permanecen presentes con ``null``.
- `request_id` lo genera el servidor; `client_request_id` se devuelve cuando
  pudo validarse (o ``null``).
- No se incluye nombre, correo, seña objetivo ni etiqueta esperada en la
  petición de predicción.
"""

from __future__ import annotations

from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from api.constants import (
    CAPTURE_PROTOCOL_VERSION,
    FRAME_COUNT,
    HTTP_SCHEMA_VERSION,
    MAX_HEIGHT,
    MAX_WIDTH,
    NUM_CLASSES,
)


# ---------------------------------------------------------------------------
# Petición: campo `metadata` (texto JSON dentro del multipart)
# ---------------------------------------------------------------------------
class PredictMetadata(BaseModel):
    """Metadatos JSON del intento de predicción.

    La API valida estructura, orden y límites; los metadatos por sí solos NO
    prueban cómo se capturó el video (eso lo cubren las pruebas físicas y de
    navegador).
    """

    model_config = ConfigDict(extra="forbid")  # rechazar campos desconocidos

    schema_version: Literal["lsc.http.v1"] = Field(
        description="Cadena fija del contrato."
    )
    client_request_id: str = Field(
        description="UUID nuevo por intento; sólo correlación, sin identidad personal."
    )
    capture_protocol_version: str = Field(
        description="Versión exacta anunciada por la configuración del servicio."
    )
    source: Literal["webcam", "authorized_fixture"] = Field(
        description="Origen de los cuadros; authorized_fixture queda marcado en la evidencia."
    )
    width: int = Field(gt=0, le=MAX_WIDTH, description="Ancho JPEG decodificado.")
    height: int = Field(gt=0, le=MAX_HEIGHT, description="Alto JPEG decodificado.")
    frame_count: Literal[6] = Field(description="Número de cuadros enviados.")
    observed_frame_count: int = Field(
        ge=FRAME_COUNT, description="N >= 6 de cuadros muestreados antes de selección."
    )
    selected_indices: list[int] = Field(
        min_length=FRAME_COUNT,
        max_length=FRAME_COUNT,
        description="Seis enteros que cumplen la fórmula de muestreo y el rango 0..N-1.",
    )
    timestamps_ms: list[float] = Field(
        min_length=FRAME_COUNT,
        max_length=FRAME_COUNT,
        description="Seis tiempos finitos estrictamente crecientes, relativos al inicio.",
    )
    capture_duration_ms: float = Field(
        gt=0, description="Duración real de la ventana; incluye los timestamps seleccionados."
    )
    mirrored: Literal[False] = Field(
        description="Debe ser false: los bytes enviados no van espejados."
    )

    @field_validator("client_request_id")
    @classmethod
    def _valid_uuid(cls, v: str) -> str:
        import uuid

        try:
            uuid.UUID(str(v))
        except (ValueError, AttributeError, TypeError) as exc:
            raise ValueError("client_request_id debe ser un UUID válido.") from exc
        return v

    @field_validator("selected_indices")
    @classmethod
    def _indices_sorted_distinct(cls, v: list[int]) -> list[int]:
        if any(i < 0 for i in v):
            raise ValueError("selected_indices no puede contener valores negativos.")
        if list(v) != sorted(v) or len(set(v)) != len(v):
            raise ValueError("selected_indices debe ser estrictamente creciente y distinto.")
        return v

    @field_validator("timestamps_ms")
    @classmethod
    def _timestamps_increasing_finite(cls, v: list[float]) -> list[float]:
        import math

        if any(not math.isfinite(t) for t in v):
            raise ValueError("timestamps_ms debe contener valores finitos.")
        if any(b <= a for a, b in zip(v, v[1:])):
            raise ValueError("timestamps_ms debe ser estrictamente creciente.")
        return v

    @model_validator(mode="after")
    def _cross_field(self) -> "PredictMetadata":
        # selected_indices dentro de 0..observed_frame_count-1
        if any(i > self.observed_frame_count - 1 for i in self.selected_indices):
            raise ValueError(
                "selected_indices fuera de rango 0..observed_frame_count-1."
            )
        # Verificar la fórmula de muestreo floor(k*(N-1)/5 + 0.5)
        n = self.observed_frame_count
        expected = [int((k * (n - 1)) / 5 + 0.5) for k in range(FRAME_COUNT)]
        if self.selected_indices != expected:
            raise ValueError(
                f"selected_indices {self.selected_indices} no cumple la fórmula "
                f"de muestreo; esperado {expected} para N={n}."
            )
        # El último timestamp debe quedar dentro de la duración declarada.
        if self.timestamps_ms[-1] > self.capture_duration_ms:
            raise ValueError(
                "El último timestamp supera capture_duration_ms."
            )
        return self


# ---------------------------------------------------------------------------
# Respuesta: sobre JSON uniforme
# ---------------------------------------------------------------------------
class TimingsMs(BaseModel):
    """Tiempos del servidor en milisegundos. ``null`` para etapas no ejecutadas."""

    model_config = ConfigDict(extra="forbid")

    decode: float | None = None
    extraction: float | None = None
    inference: float | None = None
    decision: float | None = None
    server_total: float | None = None


class PredictResponse(BaseModel):
    """Sobre JSON común a accepted / retry / error."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["lsc.http.v1"] = HTTP_SCHEMA_VERSION
    request_id: str = Field(description="UUID generado por el servidor.")
    client_request_id: str | None = Field(
        default=None, description="UUID de la solicitud o null si no se pudo validar."
    )
    status: Literal["accepted", "retry", "error"]
    label_id: int | None = Field(default=None, ge=0, le=NUM_CLASSES - 1)
    display_text: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    reason_codes: list[str] = Field(default_factory=list)
    model_version: str | None = None
    feature_spec_hash: str | None = None
    capture_protocol_version: str | None = None
    mode: Literal["real", "mock"]
    timings_ms: TimingsMs = Field(default_factory=TimingsMs)

    @model_validator(mode="after")
    def _status_invariants(self) -> "PredictResponse":
        if self.status == "accepted":
            if self.label_id is None or self.display_text is None or self.confidence is None:
                raise ValueError("accepted requiere label_id, display_text y confidence.")
        else:
            # retry / error: sin sugerencia.
            if self.label_id is not None or self.display_text is not None or self.confidence is not None:
                raise ValueError("retry/error no deben ofrecer sugerencia (label/confidence null).")
        return self


# ---------------------------------------------------------------------------
# /health y /config
# ---------------------------------------------------------------------------
class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ready: bool
    busy: bool
    mode: Literal["real", "mock"]
    api_version: str
    model_version: str | None = None
    feature_spec_hash: str | None = None
    capture_protocol_version: str | None = None


class ConfigResponse(BaseModel):
    """Configuración pública para la UI (sin rutas internas)."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["lsc.http.v1"] = HTTP_SCHEMA_VERSION
    mode: Literal["real", "mock"]
    api_version: str
    labels: list[str]
    capture_protocol: dict
    limits: dict
    capabilities: dict
    capture_protocol_version: str = CAPTURE_PROTOCOL_VERSION
