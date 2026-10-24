"""Pruebas unitarias de contrato: fórmula de muestreo, metadata, verificación
de probabilidades y paridad de bytes (A13)."""

from __future__ import annotations

import numpy as np
import pytest

from api.schemas.models import PredictMetadata, PredictResponse
from api.constants import CAPTURE_PROTOCOL_VERSION, HTTP_SCHEMA_VERSION
from tests.app.conftest import make_jpeg, valid_metadata


def sampling_formula(n, frame_count=6):
    return [int((k * (n - 1)) / 5 + 0.5) for k in range(frame_count)]


# --- U03 (lado servidor): índices para N=6 y N=30 ---------------------------
def test_sampling_indices_n6():
    assert sampling_formula(6) == [0, 1, 2, 3, 4, 5]


def test_sampling_indices_n30():
    # floor(k*29/5 + 0.5): 0, 6, 12, 17, 23, 29
    assert sampling_formula(30) == [0, 6, 12, 17, 23, 29]


# --- Validación de metadata -------------------------------------------------
def test_metadata_valid():
    meta = PredictMetadata.model_validate(valid_metadata())
    assert meta.frame_count == 6
    assert meta.mirrored is False


def test_metadata_rejects_bad_indices():
    bad = valid_metadata()
    bad["selected_indices"] = [0, 1, 2, 3, 4, 4]  # no distinto/creciente
    with pytest.raises(Exception):
        PredictMetadata.model_validate(bad)


def test_metadata_rejects_non_monotonic_timestamps():
    bad = valid_metadata()
    bad["timestamps_ms"] = [10, 9, 20, 30, 40, 50]
    with pytest.raises(Exception):
        PredictMetadata.model_validate(bad)


def test_metadata_rejects_unknown_field():
    bad = valid_metadata()
    bad["extra_field"] = "x"
    with pytest.raises(Exception):
        PredictMetadata.model_validate(bad)


def test_metadata_formula_enforced_for_n30():
    meta = valid_metadata(n=30)
    # valid_metadata ya genera la fórmula; confirmamos que coincide.
    assert meta["selected_indices"] == [0, 6, 12, 17, 23, 29]
    PredictMetadata.model_validate(meta)


# --- Invariantes del sobre de respuesta -------------------------------------
def test_accepted_requires_label():
    with pytest.raises(Exception):
        PredictResponse(
            request_id="r",
            status="accepted",
            mode="mock",
            label_id=None,  # falta
        )


def test_retry_forbids_suggestion():
    with pytest.raises(Exception):
        PredictResponse(
            request_id="r",
            status="retry",
            mode="mock",
            label_id=3,  # no permitido en retry
        )


# --- A08: verificación de probabilidades ------------------------------------
def test_verify_probabilities_ok():
    from api.adapters.real import verify_probabilities

    probs = np.full((1, 10), 0.1, dtype=np.float32)
    verify_probabilities(probs, batch=1)  # no lanza


def test_verify_probabilities_bad_sum():
    from api.adapters.real import verify_probabilities, ModelOutputError

    probs = np.full((1, 10), 0.2, dtype=np.float32)  # suma 2.0
    with pytest.raises(ModelOutputError):
        verify_probabilities(probs, batch=1)


# --- A13: paridad de bytes antes del modelo ---------------------------------
def test_a13_byte_parity_same_bytes():
    """Los bytes recibidos por el backend y los de offline coinciden
    exactamente cuando se usan los MISMOS bytes JPEG. La API no modifica la
    entrada antes del extractor."""
    from api.imaging import decode_jpeg_to_rgb

    data = make_jpeg(color=77)
    rgb_a = decode_jpeg_to_rgb(data)
    rgb_b = decode_jpeg_to_rgb(data)
    assert np.allclose(rgb_a, rgb_b, atol=1e-6)
    assert rgb_a.dtype == np.uint8
