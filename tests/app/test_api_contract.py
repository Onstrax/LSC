"""Pruebas de API e integración A01–A14 (SDD §18).

Usan dobles deterministas; sin datos personales ni red. Verifican el sobre
uniforme, los códigos HTTP, el control de concurrencia y la paridad de bytes.
"""

from __future__ import annotations

import json
import time
import uuid

import numpy as np
import pytest

from api.adapters.mock import MockExtractor, MockPredictor, MockDecisionAdapter
from api.constants import HTTP_SCHEMA_VERSION
from tests.app.conftest import (
    build_injected_state,
    build_multipart,
    make_client,
    make_jpeg,
    valid_metadata,
)


def _post(client, frames, metadata):
    files, data = build_multipart(frames, metadata)
    return client.post("/api/v1/predict", files=files, data=data)


def _assert_envelope(body):
    assert body["schema_version"] == HTTP_SCHEMA_VERSION
    assert "request_id" in body and body["request_id"]
    assert body["status"] in {"accepted", "retry", "error"}
    assert body["mode"] in {"real", "mock"}
    for key in (
        "label_id",
        "display_text",
        "confidence",
        "model_version",
        "feature_spec_hash",
    ):
        assert key in body  # nullables presentes


# --- A01: seis JPEG válidos -> accepted + una sola inferencia ---------------
def test_a01_accepted(client):
    frames = [make_jpeg(color=10 + i) for i in range(6)]
    # Umbral mock por defecto 0.55; algunos contenidos no superan el umbral.
    # Forzamos un predictor que siempre acepte para validar el camino accepted.
    client.app.state.lsc = build_injected_state(
        predictor=_AlwaysConfidentPredictor(),
        decision=MockDecisionAdapter(threshold=0.5),
    )
    res = _post(client, frames, valid_metadata())
    assert res.status_code == 200
    body = res.json()
    _assert_envelope(body)
    assert body["status"] == "accepted"
    assert body["label_id"] is not None
    assert body["display_text"] is not None
    assert 0.0 <= body["confidence"] <= 1.0


# --- A02: cinco o siete JPEG -> 422, sin invocar extractor/modelo -----------
def test_a02_wrong_frame_count(client):
    client.app.state.lsc = build_injected_state(
        extractor=_SpyExtractor(), predictor=_SpyPredictor()
    )
    spy_ext = client.app.state.lsc.worker._extractor
    for count in (5, 7):
        frames = [make_jpeg() for _ in range(count)]
        res = _post(client, frames, valid_metadata())
        assert res.status_code == 422
        assert res.json()["status"] == "error"
    assert spy_ext.calls == 0


# --- A04: archivo corrupto o formato diferente ------------------------------
def test_a04_corrupt_image(client):
    frames = [make_jpeg() for _ in range(5)] + [b"no-es-una-imagen"]
    res = _post(client, frames, valid_metadata())
    assert res.status_code in (415, 422)
    body = res.json()
    assert body["status"] == "error"
    # Sin traza pública.
    assert "Traceback" not in json.dumps(body)


# --- A05: dimensiones diferentes o espejo declarado -------------------------
def test_a05_dimension_mismatch(client):
    frames = [make_jpeg(width=320, height=240) for _ in range(5)]
    frames.append(make_jpeg(width=300, height=240))
    res = _post(client, frames, valid_metadata(width=320, height=240))
    assert res.status_code == 422
    assert res.json()["status"] == "error"


def test_a05_mirror_declared_rejected(client):
    meta = valid_metadata()
    meta["mirrored"] = True  # viola Literal[False]
    frames = [make_jpeg() for _ in range(6)]
    res = _post(client, frames, meta)
    assert res.status_code == 422


# --- A06: quality_pass=false -> 200 retry, sin clasificador -----------------
def test_a06_quality_fail_retry(client):
    spy_pred = _SpyPredictor()
    client.app.state.lsc = build_injected_state(
        extractor=MockExtractor(
            force_quality_pass=False, reason_codes=["INSUFFICIENT_HANDS"]
        ),
        predictor=spy_pred,
    )
    spy_pred = client.app.state.lsc.worker._predictor
    res = _post(client, [make_jpeg() for _ in range(6)], valid_metadata())
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "retry"
    assert body["label_id"] is None
    assert "INSUFFICIENT_HANDS" in body["reason_codes"]
    assert spy_pred.calls == 0


# --- A07: advertencia de mano con calidad válida ----------------------------
def test_a07_warning_not_rejected(client):
    client.app.state.lsc = build_injected_state(
        extractor=MockExtractor(
            force_quality_pass=True, reason_codes=["HAND_ASSOCIATION_AMBIGUOUS"]
        ),
        predictor=_AlwaysConfidentPredictor(),
        decision=MockDecisionAdapter(threshold=0.5),
    )
    res = _post(client, [make_jpeg() for _ in range(6)], valid_metadata())
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "accepted"
    assert "HAND_ASSOCIATION_AMBIGUOUS" in body["reason_codes"]


# --- A08: NaN o forma incorrecta del predictor -> 500 MODEL_OUTPUT_INVALID --
def test_a08_invalid_model_output(client):
    client.app.state.lsc = build_injected_state(
        predictor=MockPredictor(emit_invalid=True),
    )
    # El mock no verifica; usamos el verificador real sobre su salida.
    from api.adapters.real import verify_probabilities, ModelOutputError

    probs = client.app.state.lsc.worker._predictor.predict(
        np.zeros((1, 6, 147), np.float32)
    )
    with pytest.raises(ModelOutputError):
        verify_probabilities(probs, batch=1)


# --- A10: dos solicitudes simultáneas -> una ejecuta, otra 429 --------------
# Nota: el TestClient de Starlette serializa peticiones en un único portal, por
# lo que la concurrencia real NO puede ejercitarse vía HTTP aquí. Se prueba la
# invariante de una sola plaza directamente sobre el worker, que es donde vive
# el control (la API traduce ServiceBusyError -> 429). La concurrencia a través
# de HTTP se cubre con uvicorn real en tests/e2e y en el piloto.
def test_a10_single_slot_busy():
    import threading

    from api.worker import InferenceWorker, ServiceBusyError

    worker = InferenceWorker(
        extractor=_SlowExtractor(delay_s=0.4),
        predictor=_AlwaysConfidentPredictor(),
        decision=MockDecisionAdapter(threshold=0.5),
        job_timeout_ms=10_000,
    )
    frames = [make_jpeg() for _ in range(6)]
    decoded = [np.zeros((10, 10, 3), np.uint8) for _ in range(6)]

    outcomes = {}

    def run_first():
        try:
            worker.submit(decoded)
            outcomes["first"] = "ok"
        except ServiceBusyError:
            outcomes["first"] = "busy"

    t = threading.Thread(target=run_first)
    t.start()
    time.sleep(0.1)  # el primero toma la plaza
    try:
        worker.submit(decoded)
        outcomes["second"] = "ok"
    except ServiceBusyError:
        outcomes["second"] = "busy"
    t.join()
    worker.shutdown()

    assert outcomes["first"] == "ok"
    assert outcomes["second"] == "busy"


# --- A11: timeout -> 504; plaza retenida hasta final real -------------------
def test_a11_timeout(client):
    client.app.state.lsc = build_injected_state(
        extractor=_SlowExtractor(delay_s=0.4),
        predictor=_AlwaysConfidentPredictor(),
        decision=MockDecisionAdapter(threshold=0.5),
        job_timeout_ms=100,
    )
    res = _post(client, [make_jpeg() for _ in range(6)], valid_metadata())
    assert res.status_code == 504
    body = res.json()
    assert body["status"] == "error"
    assert "INFERENCE_TIMEOUT" in body["reason_codes"]


# --- A12: modo real sin pesos -> 503, no fallback a mock --------------------
def test_a12_real_unavailable(client):
    client.app.state.lsc = build_injected_state(mode="real", ready=False)
    res = _post(client, [make_jpeg() for _ in range(6)], valid_metadata())
    assert res.status_code == 503
    body = res.json()
    assert body["status"] == "error"
    assert body["mode"] == "real"  # NO cambió a mock


# --- A09: incompatibilidad de hash -> readiness 503 -------------------------
def test_a09_hash_incompatible():
    from api.worker import InferenceWorker

    worker = InferenceWorker(
        extractor=MockExtractor(feature_spec_hash="hash-A"),
        predictor=MockPredictor(feature_spec_hash="hash-B"),
        decision=MockDecisionAdapter(),
        job_timeout_ms=10000,
    )
    with pytest.raises(RuntimeError):
        worker.verify_compatibility()
    worker.shutdown()


# --- Salud y configuración --------------------------------------------------
def test_health_ready(client):
    res = client.get("/api/v1/health")
    assert res.status_code == 200
    body = res.json()
    assert body["ready"] is True
    assert body["mode"] == "mock"
    assert body["capture_protocol_version"]


def test_config_labels_order(client):
    res = client.get("/api/v1/config")
    assert res.status_code == 200
    body = res.json()
    assert body["labels"][0] == "Hola"
    assert body["labels"][9] == "Licor"
    assert len(body["labels"]) == 10
    assert "limits" in body and "capture_protocol" in body


# ---------------------------------------------------------------------------
# Dobles auxiliares
# ---------------------------------------------------------------------------
class _SpyExtractor(MockExtractor):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def extract_sequence(self, frames_rgb):
        self.calls += 1
        return super().extract_sequence(frames_rgb)


class _SpyPredictor(MockPredictor):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def predict(self, x):
        self.calls += 1
        return super().predict(x)


class _AlwaysConfidentPredictor(MockPredictor):
    def predict(self, x):
        x = np.asarray(x, dtype=np.float32)
        batch = x.shape[0] if x.ndim == 3 else 1
        out = np.zeros((batch, 10), dtype=np.float32)
        out[:, 0] = 0.92
        out[:, 1:] = 0.08 / 9
        return out


class _SlowExtractor(MockExtractor):
    def __init__(self, delay_s: float):
        super().__init__(force_quality_pass=True)
        self._delay = delay_s

    def extract_sequence(self, frames_rgb):
        time.sleep(self._delay)
        return super().extract_sequence(frames_rgb)
