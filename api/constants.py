"""Constantes del contrato HTTP y del protocolo de captura.

Fuente única de verdad para los valores que la API y la interfaz comparten.
El *orden canónico de etiquetas* NO se define aquí como texto de visualización:
el texto proviene de `labels.json` (Acosta) a través de `GET /api/v1/config`.
Aquí sólo se fija el contrato (versiones, límites, códigos).
"""

from __future__ import annotations

# --- Versiones de contrato -------------------------------------------------
HTTP_SCHEMA_VERSION = "lsc.http.v1"
"""Valor fijo de `schema_version` en petición y respuesta."""

# Versión del protocolo de captura. Puede cambiar con Ortiz y Acosta durante el
# desarrollo (antes del 18 de noviembre de 2026). Cualquier cambio se refleja
# aquí y se aplica por igual al piloto y al servicio.
CAPTURE_PROTOCOL_VERSION = "lsc.capture.v1"

# --- Forma de la secuencia -------------------------------------------------
FRAME_COUNT = 6
"""Número de cuadros por intento (seis imágenes ordenadas)."""

FEATURE_DIM = 147
"""Valores por instante: 49 puntos x (x, y, presencia)."""

SEQUENCE_SHAPE = (FRAME_COUNT, FEATURE_DIM)  # (6, 147)
NUM_CLASSES = 10

# --- Dimensiones de imagen -------------------------------------------------
MAX_WIDTH = 640
MAX_HEIGHT = 480
MAX_PIXELS = MAX_WIDTH * MAX_HEIGHT  # 307_200

# --- Límites de tamaño (bytes) ---------------------------------------------
MAX_BODY_BYTES = 6 * 1024 * 1024  # 6 MiB cuerpo multipart completo
MAX_JPEG_BYTES = 512 * 1024       # 512 KiB por JPEG
MAX_METADATA_BYTES = 16 * 1024    # 16 KiB de metadata

# --- Plazos (milisegundos) -------------------------------------------------
SERVER_JOB_TIMEOUT_MS = 10_000  # plazo del trabajo de servidor
UI_WAIT_TIMEOUT_MS = 12_000     # espera de la interfaz
# Meta p95 < 2 s es INDEPENDIENTE de estos límites de recuperación.
P95_TARGET_MS = 2_000

# --- Protocolo de captura (ms) ---------------------------------------------
CAPTURE_PREP_MS = 2_000      # preparación visible
CAPTURE_WINDOW_MS = 1_000    # ventana de captura
CAPTURE_MAX_FPS = 30         # muestras por segundo máximas

# --- Modos -----------------------------------------------------------------
MODE_REAL = "real"
MODE_MOCK = "mock"

# --- Estados de respuesta --------------------------------------------------
STATUS_ACCEPTED = "accepted"
STATUS_RETRY = "retry"
STATUS_ERROR = "error"

# --- Fuente de la captura --------------------------------------------------
SOURCE_WEBCAM = "webcam"
SOURCE_FIXTURE = "authorized_fixture"

# --- Códigos de razón ------------------------------------------------------
# Códigos de CALIDAD del extractor (propiedad de Ortiz). La API los respeta;
# no los inventa ni los convierte en rechazo por cuenta propia.
QUALITY_REASON_CODES = frozenset(
    {
        "INVALID_FRAME_COUNT",
        "INVALID_IMAGE",
        "SIZE_MISMATCH",
        "MULTIPLE_PEOPLE",
        "INSUFFICIENT_POSE",
        "INSUFFICIENT_HANDS",
        "HAND_ASSOCIATION_AMBIGUOUS",  # puede ser advertencia si quality_pass=true
        "GEOMETRY_OUTLIER",
        "CONFIG_MISMATCH",
    }
)

# Advertencias no bloqueantes que NO implican rechazo por sí solas.
NON_BLOCKING_WARNINGS = frozenset({"HAND_ASSOCIATION_AMBIGUOUS"})

# Códigos de la API (petición / infraestructura).
API_REASON_CODES = frozenset(
    {
        "INVALID_METADATA",
        "MALFORMED_MULTIPART",
        "BODY_TOO_LARGE",
        "FILE_TOO_LARGE",
        "UNSUPPORTED_MEDIA_TYPE",
        "CONTRACT_VIOLATION",
        "DIMENSION_MISMATCH",
        "MIRROR_DECLARED",
        "SERVICE_BUSY",
        "MODEL_UNAVAILABLE",
        "CONFIG_INCOMPATIBLE",
        "INFERENCE_TIMEOUT",
        "MODEL_OUTPUT_INVALID",
        "INTERNAL_ERROR",
        "LOW_CONFIDENCE",  # rechazo por umbral (política de Acosta)
    }
)

# Tolerancia para verificar que las probabilidades suman ~1.
PROB_SUM_ATOL = 1e-4
