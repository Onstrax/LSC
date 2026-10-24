"""Utilidades y fixtures de pruebas de API con dobles deterministas.

No requieren dataset, red ni modelos reales. Construyen un TestClient sobre la
aplicación FastAPI en modo mock, con posibilidad de inyectar dobles específicos
(extractor/predictor/decisión) para ejercitar los tres estados.
"""

from __future__ import annotations

import io
import json
import uuid

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from api.config import load_settings
from api.constants import CAPTURE_PROTOCOL_VERSION, HTTP_SCHEMA_VERSION
from api.service import build_state
from api.worker import InferenceWorker


# ---------------------------------------------------------------------------
# Generación de imágenes y metadata válidas
# ---------------------------------------------------------------------------
def make_jpeg(width: int = 320, height: int = 240, color: int = 100) -> bytes:
    """Crea un JPEG válido de dimensiones dadas."""
    arr = np.full((height, width, 3), color, dtype=np.uint8)
    # Un gradiente ligero para variar bytes entre colores.
    arr[:, :, 0] = (np.arange(width) % 256).astype(np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def valid_metadata(width: int = 320, height: int = 240, n: int = 6) -> dict:
    """Metadata contractual con fórmula de muestreo correcta para N=n."""
    selected = [int((k * (n - 1)) / 5 + 0.5) for k in range(6)]
    timestamps = [float(i * 150 + 10) for i in range(6)]  # estrictamente creciente
    return {
        "schema_version": HTTP_SCHEMA_VERSION,
        "client_request_id": str(uuid.uuid4()),
        "capture_protocol_version": CAPTURE_PROTOCOL_VERSION,
        "source": "authorized_fixture",
        "width": width,
        "height": height,
        "frame_count": 6,
        "observed_frame_count": n,
        "selected_indices": selected,
        "timestamps_ms": timestamps,
        "capture_duration_ms": 1000.0,
        "mirrored": False,
    }


def build_multipart(frames: list[bytes], metadata: dict) -> tuple[list, dict]:
    """Devuelve (files, data) para TestClient.post."""
    files = [
        ("frames", (f"frame_{i}.jpg", data, "image/jpeg"))
        for i, data in enumerate(frames)
    ]
    data = {"metadata": json.dumps(metadata)}
    return files, data


# ---------------------------------------------------------------------------
# Cliente con worker inyectable
# ---------------------------------------------------------------------------
def build_injected_state(
    *,
    extractor=None,
    predictor=None,
    decision=None,
    mode: str = "mock",
    ready: bool = True,
    job_timeout_ms: int | None = None,
):
    """Construye un ServiceState con un worker de dobles ya inicializado."""
    from api.adapters.mock import MockDecisionAdapter, MockExtractor, MockPredictor
    from api.config import load_labels

    extractor = extractor if extractor is not None else MockExtractor()
    predictor = predictor if predictor is not None else MockPredictor()
    decision = decision if decision is not None else MockDecisionAdapter()

    import os

    os.environ["LSC_MODE"] = mode
    # El TestClient usa host "testserver"; permitirlo sin Origin en pruebas.
    os.environ["LSC_ALLOWED_HOSTS"] = "localhost,127.0.0.1,testserver"
    settings = load_settings()
    state = build_state(settings)
    state.mode = mode
    state.labels = load_labels(settings) if mode == "mock" else []

    if ready:
        worker = InferenceWorker(
            extractor=extractor,
            predictor=predictor,
            decision=decision,
            job_timeout_ms=job_timeout_ms or settings.server_job_timeout_ms,
        )
        worker.mark_ready()
        state.worker = worker
    else:
        state.worker = None
        state.unavailable_reason = "forzado no disponible para prueba"
    return state


def make_client(**kwargs) -> TestClient:
    """TestClient cuyo estado se inyecta tras el arranque del lifespan."""
    import api.main as main_module

    state = build_injected_state(**kwargs)
    client = TestClient(main_module.app)
    client.__enter__()  # dispara lifespan (carga por defecto, luego la pisamos)
    client.app.state.lsc = state
    return client


@pytest.fixture()
def client():
    c = make_client()
    try:
        yield c
    finally:
        state = c.app.state.lsc
        if state.worker is not None:
            state.worker.shutdown()
        c.__exit__(None, None, None)
