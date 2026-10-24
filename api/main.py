"""Aplicación FastAPI del módulo de Juan (contrato `lsc.http.v1`).

Sirve la interfaz estática y la API desde el MISMO origen. Implementa:
- POST /api/v1/predict  (multipart/form-data, seis JPEG + metadata JSON)
- GET  /api/v1/health
- GET  /api/v1/config
- GET  /api/v1/live     (liveness simple)

No usa CORS abierto: valida Host/Origin para el origen local permitido.
"""

from __future__ import annotations

import json
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Form, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from api import __version__ as API_VERSION
from api.config import capture_protocol_dict, limits_dict, load_settings
from api.constants import (
    CAPTURE_PROTOCOL_VERSION,
    FRAME_COUNT,
    HTTP_SCHEMA_VERSION,
    MAX_BODY_BYTES,
    MAX_JPEG_BYTES,
    MAX_METADATA_BYTES,
    MODE_MOCK,
    STATUS_ACCEPTED,
    STATUS_RETRY,
)
from api.errors import build_envelope, error_response, new_request_id
from api.imaging import ImageError, decode_sequence
from api.schemas.models import (
    ConfigResponse,
    HealthResponse,
    PredictMetadata,
)
from api.service import build_state, initialize_worker
from api.worker import InferenceTimeoutError, ServiceBusyError

APP_DIR = Path(__file__).resolve().parent.parent / "app"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carga y libera recursos (SDD §12).

    Inicializa el worker en estado loading; publica readiness sólo tras
    verificar artefactos y política. No carga por petición.
    """
    settings = load_settings()
    state = build_state(settings)
    app.state.lsc = state
    # Carga síncrona en el arranque (un solo worker dedicado).
    initialize_worker(state)
    try:
        yield
    finally:
        if state.worker is not None:
            state.worker.shutdown()


app = FastAPI(
    title="LSC · Reconocimiento de señas aisladas (módulo de Juan)",
    version=API_VERSION,
    lifespan=lifespan,
    description=(
        "Contrato lsc.http.v1. Aplicación web local que captura una seña "
        "aislada de LSC y devuelve una sugerencia de etiqueta. El "
        "reconocimiento proviene del extractor (Ortiz) y el modelo (Acosta)."
    ),
)


# ---------------------------------------------------------------------------
# Middleware: validación de Host/Origin para el origen local (sin CORS abierto)
# ---------------------------------------------------------------------------
@app.middleware("http")
async def origin_guard(request: Request, call_next):
    state = request.app.state.lsc
    settings = state.settings
    origin = request.headers.get("origin")
    # El uso de CLI local sin Origin se permite (documentado para pruebas).
    if origin is not None and origin not in settings.allowed_origins:
        return JSONResponse(
            status_code=403,
            content={"detail": "Origin no permitido para el servicio local."},
        )
    # Validación de Host (cabecera host puede incluir puerto).
    host_header = (request.headers.get("host") or "").split(":")[0]
    if host_header and host_header not in settings.allowed_hosts:
        return JSONResponse(
            status_code=403,
            content={"detail": "Host no permitido para el servicio local."},
        )
    return await call_next(request)


# ---------------------------------------------------------------------------
# Salud y configuración
# ---------------------------------------------------------------------------
@app.get("/api/v1/live")
async def live() -> dict:
    """Liveness: confirma que el proceso está activo."""
    return {"alive": True, "api_version": API_VERSION}


@app.get("/api/v1/health")
async def health(request: Request) -> JSONResponse:
    state = request.app.state.lsc
    worker = state.worker
    ready = worker is not None and worker.ready
    busy = bool(worker.busy) if worker is not None else False
    payload = HealthResponse(
        ready=ready,
        busy=busy,
        mode=state.mode_label,
        api_version=API_VERSION,
        model_version=(worker.model_version if worker else None),
        feature_spec_hash=(worker.feature_spec_hash if worker else None),
        capture_protocol_version=CAPTURE_PROTOCOL_VERSION,
    )
    # 200 si la configuración está preparada (incluso busy); 503 si cargando,
    # degradada o incompatible.
    http_status = 200 if ready else 503
    return JSONResponse(status_code=http_status, content=payload.model_dump())


@app.get("/api/v1/config", response_model=ConfigResponse)
async def config(request: Request) -> ConfigResponse:
    state = request.app.state.lsc
    worker = state.worker
    return ConfigResponse(
        schema_version=HTTP_SCHEMA_VERSION,
        mode=state.mode_label,
        api_version=API_VERSION,
        labels=list(state.labels),
        capture_protocol=capture_protocol_dict(),
        limits=limits_dict(),
        capabilities={
            "requires_secure_context": True,
            "requires_request_video_frame_callback": True,
            "single_inference_slot": True,
            "ui_wait_timeout_ms": state.settings.ui_wait_timeout_ms,
        },
        capture_protocol_version=CAPTURE_PROTOCOL_VERSION,
    )


# ---------------------------------------------------------------------------
# Predicción
# ---------------------------------------------------------------------------
@app.post("/api/v1/predict")
async def predict(
    request: Request,
    frames: list[UploadFile] = [],  # noqa: B006 - FastAPI maneja el default
    metadata: str = Form(...),
) -> JSONResponse:
    state = request.app.state.lsc
    settings = state.settings
    mode = state.mode_label
    server_t0 = time.perf_counter()

    # 0) Límite de cuerpo por Content-Length anunciado (defensivo; el límite
    #    real al leer el flujo se aplica por archivo más abajo).
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            if int(content_length) > MAX_BODY_BYTES:
                return error_response(
                    http_status=413, mode=mode, reason_codes=["BODY_TOO_LARGE"]
                )
        except ValueError:
            pass

    # 1) Validar metadata (JSON legible + contrato).
    if len(metadata.encode("utf-8")) > MAX_METADATA_BYTES:
        return error_response(http_status=413, mode=mode, reason_codes=["BODY_TOO_LARGE"])
    try:
        meta_dict = json.loads(metadata)
    except json.JSONDecodeError:
        return error_response(
            http_status=400, mode=mode, reason_codes=["INVALID_METADATA"]
        )
    try:
        meta = PredictMetadata.model_validate(meta_dict)
    except ValidationError:
        # No se pudo validar -> client_request_id null.
        return error_response(
            http_status=422, mode=mode, reason_codes=["CONTRACT_VIOLATION"]
        )

    client_request_id = meta.client_request_id

    # mirrored=False y coherencia de protocolo ya validados por el modelo.
    if meta.capture_protocol_version != CAPTURE_PROTOCOL_VERSION:
        return error_response(
            http_status=422,
            mode=mode,
            reason_codes=["CONTRACT_VIOLATION"],
            client_request_id=client_request_id,
        )

    # 2) Validar número de archivos (exactamente seis).
    if len(frames) != FRAME_COUNT:
        return error_response(
            http_status=422,
            mode=mode,
            reason_codes=["CONTRACT_VIOLATION"],
            client_request_id=client_request_id,
        )

    # 3) Leer y validar tamaño de cada archivo antes de grandes buffers.
    raw_frames: list[bytes] = []
    for uf in frames:
        # Validación de Content-Type declarado (no basta por sí sola).
        ctype = (uf.content_type or "").lower()
        if ctype and "jpeg" not in ctype and "jpg" not in ctype:
            await _close_uploads(frames)
            return error_response(
                http_status=415,
                mode=mode,
                reason_codes=["UNSUPPORTED_MEDIA_TYPE"],
                client_request_id=client_request_id,
            )
        data = await _read_limited(uf, MAX_JPEG_BYTES)
        if data is None:
            await _close_uploads(frames)
            return error_response(
                http_status=413,
                mode=mode,
                reason_codes=["FILE_TOO_LARGE"],
                client_request_id=client_request_id,
            )
        raw_frames.append(data)

    total = sum(len(d) for d in raw_frames) + len(metadata.encode("utf-8"))
    if total > MAX_BODY_BYTES:
        await _close_uploads(frames)
        return error_response(
            http_status=413,
            mode=mode,
            reason_codes=["BODY_TOO_LARGE"],
            client_request_id=client_request_id,
        )

    # 4) Decodificar JPEG a RGB y verificar dimensiones.
    decode_t0 = time.perf_counter()
    try:
        decoded = decode_sequence(raw_frames, declared_wh=(meta.width, meta.height))
    except ImageError as exc:
        await _close_uploads(frames)
        http_status = 415 if exc.reason_code == "UNSUPPORTED_MEDIA_TYPE" else 422
        return error_response(
            http_status=http_status,
            mode=mode,
            reason_codes=[exc.reason_code],
            client_request_id=client_request_id,
            capture_protocol_version=CAPTURE_PROTOCOL_VERSION,
        )
    decode_ms = (time.perf_counter() - decode_t0) * 1000.0
    await _close_uploads(frames)

    # 5) Verificar disponibilidad del servicio (modo real sin artefactos -> 503).
    worker = state.worker
    if worker is None or not worker.ready:
        code = "CONFIG_INCOMPATIBLE" if state.unavailable_reason else "MODEL_UNAVAILABLE"
        return error_response(
            http_status=503,
            mode=mode,
            reason_codes=[code],
            client_request_id=client_request_id,
            capture_protocol_version=CAPTURE_PROTOCOL_VERSION,
        )

    # 6) Ejecutar en el worker (una sola plaza).
    try:
        result = worker.submit(decoded)
    except ServiceBusyError:
        return error_response(
            http_status=429,
            mode=mode,
            reason_codes=["SERVICE_BUSY"],
            client_request_id=client_request_id,
            capture_protocol_version=CAPTURE_PROTOCOL_VERSION,
            headers={"Retry-After": "1"},
        )
    except InferenceTimeoutError:
        return error_response(
            http_status=504,
            mode=mode,
            reason_codes=["INFERENCE_TIMEOUT"],
            client_request_id=client_request_id,
            capture_protocol_version=CAPTURE_PROTOCOL_VERSION,
        )
    except Exception:
        # Salida inválida del modelo u otra excepción inesperada.
        from api.adapters.real import ModelOutputError

        import sys

        exc_type = sys.exc_info()[0]
        reason = (
            "MODEL_OUTPUT_INVALID"
            if exc_type is not None and issubclass(exc_type, (ModelOutputError, ValueError))
            else "INTERNAL_ERROR"
        )
        return error_response(
            http_status=500,
            mode=mode,
            reason_codes=[reason],
            client_request_id=client_request_id,
            capture_protocol_version=CAPTURE_PROTOCOL_VERSION,
        )

    # 7) Serializar resultado.
    timings = dict(result.timings_ms)
    timings["decode"] = decode_ms
    timings["server_total"] = (time.perf_counter() - server_t0) * 1000.0

    if not result.quality_pass:
        envelope = build_envelope(
            status=STATUS_RETRY,
            mode=mode,
            client_request_id=client_request_id,
            reason_codes=result.reason_codes,
            model_version=result.model_version,
            feature_spec_hash=result.feature_spec_hash,
            capture_protocol_version=CAPTURE_PROTOCOL_VERSION,
            timings_ms=timings,
        )
        return JSONResponse(status_code=200, content=envelope.model_dump())

    if result.accepted and result.label_id is not None:
        display_text = state.labels[result.label_id]
        envelope = build_envelope(
            status=STATUS_ACCEPTED,
            mode=mode,
            client_request_id=client_request_id,
            label_id=result.label_id,
            display_text=display_text,
            confidence=result.confidence,
            reason_codes=result.reason_codes,
            model_version=result.model_version,
            feature_spec_hash=result.feature_spec_hash,
            capture_protocol_version=CAPTURE_PROTOCOL_VERSION,
            timings_ms=timings,
        )
        return JSONResponse(status_code=200, content=envelope.model_dump())

    # Rechazo por confianza/política -> retry.
    envelope = build_envelope(
        status=STATUS_RETRY,
        mode=mode,
        client_request_id=client_request_id,
        reason_codes=result.reason_codes or ["LOW_CONFIDENCE"],
        model_version=result.model_version,
        feature_spec_hash=result.feature_spec_hash,
        capture_protocol_version=CAPTURE_PROTOCOL_VERSION,
        timings_ms=timings,
    )
    return JSONResponse(status_code=200, content=envelope.model_dump())


async def _read_limited(uf: UploadFile, limit: int) -> bytes | None:
    """Lee un UploadFile aplicando un límite; devuelve None si lo excede."""
    chunks: list[bytes] = []
    size = 0
    while True:
        chunk = await uf.read(64 * 1024)
        if not chunk:
            break
        size += len(chunk)
        if size > limit:
            return None
        chunks.append(chunk)
    return b"".join(chunks)


async def _close_uploads(frames: list[UploadFile]) -> None:
    for uf in frames:
        try:
            await uf.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Estáticos (interfaz) — montados en el mismo origen
# ---------------------------------------------------------------------------
@app.get("/")
async def index() -> FileResponse:
    return FileResponse(APP_DIR / "index.html")


if APP_DIR.exists():
    app.mount("/app", StaticFiles(directory=str(APP_DIR), html=True), name="app")
    # Servir también los recursos estáticos en la raíz (styles.css, js/).
    app.mount("/static", StaticFiles(directory=str(APP_DIR)), name="static")
