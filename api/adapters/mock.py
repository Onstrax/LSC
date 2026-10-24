"""Dobles de prueba (mock) de extractor, predictor y decisión.

Propósito: probar INTEGRACIÓN, no medir reconocimiento. Son deterministas y no
dependen de pesos, dataset ni red. Se activan sólo en desarrollo con
``LSC_MODE=mock`` y la interfaz muestra el rótulo persistente
"Modo de prueba · respuestas simuladas".

Permiten forzar los tres estados (accepted / retry / error) de forma controlada
para las pruebas A01–A14.
"""

from __future__ import annotations

import hashlib

import numpy as np

from api.adapters.types import DecisionResult, FeatureResult
from api.constants import FEATURE_DIM, FRAME_COUNT, NUM_CLASSES

MOCK_FEATURE_SPEC_HASH = "mock-feature-spec-0000"
MOCK_MODEL_VERSION = "mock-model-0.0.0"
MOCK_POLICY_VERSION = "mock-policy-0.0.0"


def _seed_from_frames(frames_rgb: list[np.ndarray]) -> int:
    """Semilla determinista derivada del contenido de los cuadros."""
    h = hashlib.sha256()
    for f in frames_rgb:
        h.update(np.ascontiguousarray(f).tobytes())
    return int.from_bytes(h.digest()[:4], "big")


class MockExtractor:
    """Extractor simulado. Devuelve una matriz contractual (6,147) o un rechazo.

    Configurable para pruebas:
    - ``force_quality_pass=False`` -> rechazo de calidad con ``reason_codes``.
    - ``reason_codes`` -> lista devuelta (quality_pass se infiere si no se fuerza).
    """

    def __init__(
        self,
        *,
        force_quality_pass: bool | None = None,
        reason_codes: list[str] | None = None,
        feature_spec_hash: str = MOCK_FEATURE_SPEC_HASH,
    ) -> None:
        self._force_quality_pass = force_quality_pass
        self._reason_codes = list(reason_codes or [])
        self.feature_spec_hash = feature_spec_hash

    def extract_sequence(self, frames_rgb: list[np.ndarray]) -> FeatureResult:
        import time

        t0 = time.perf_counter()
        if len(frames_rgb) != FRAME_COUNT:
            # Esto no debería ocurrir: la API valida antes. Defensivo.
            return FeatureResult(
                x=np.zeros((FRAME_COUNT, FEATURE_DIM), dtype=np.float32),
                quality_pass=False,
                reason_codes=["INVALID_FRAME_COUNT"],
                valid_shoulders=False,
                frames_with_hand=0,
                feature_spec_hash=self.feature_spec_hash,
                elapsed_ms=(time.perf_counter() - t0) * 1000.0,
            )

        seed = _seed_from_frames(frames_rgb)
        rng = np.random.default_rng(seed)
        x = rng.standard_normal((FRAME_COUNT, FEATURE_DIM)).astype(np.float32)

        quality_pass = (
            self._force_quality_pass
            if self._force_quality_pass is not None
            else not any(
                c for c in self._reason_codes if c not in {"HAND_ASSOCIATION_AMBIGUOUS"}
            )
        )
        return FeatureResult(
            x=x,
            quality_pass=quality_pass,
            reason_codes=list(self._reason_codes),
            valid_shoulders=True,
            frames_with_hand=FRAME_COUNT,
            feature_spec_hash=self.feature_spec_hash,
            elapsed_ms=(time.perf_counter() - t0) * 1000.0,
        )


class MockPredictor:
    """Predictor simulado determinista.

    Devuelve probabilidades (B,10) reproducibles a partir del contenido de ``x``
    mediante softmax sobre logits derivados de un hash. Puede configurarse para
    producir salidas inválidas (NaN / forma incorrecta) y ejercitar A08.
    """

    def __init__(
        self,
        *,
        emit_invalid: bool = False,
        feature_spec_hash: str = MOCK_FEATURE_SPEC_HASH,
        model_version: str = MOCK_MODEL_VERSION,
    ) -> None:
        self._emit_invalid = emit_invalid
        self.feature_spec_hash = feature_spec_hash
        self.model_version = model_version

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self._emit_invalid:
            # Salida inválida para probar MODEL_OUTPUT_INVALID (A08).
            bad = np.full((x.shape[0], NUM_CLASSES), np.nan, dtype=np.float32)
            return bad

        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 2:
            x = x[None, ...]
        batch = x.shape[0]
        out = np.empty((batch, NUM_CLASSES), dtype=np.float32)
        for i in range(batch):
            # sha512 da 64 bytes -> 16 uint32; tomamos NUM_CLASSES=10.
            digest = hashlib.sha512(np.ascontiguousarray(x[i]).tobytes()).digest()
            raw = np.frombuffer(digest[: NUM_CLASSES * 4], dtype=np.uint32)
            logits = (raw.astype(np.float64) / np.iinfo(np.uint32).max) * 6.0 - 3.0
            logits -= logits.max()
            e = np.exp(logits)
            out[i] = (e / e.sum()).astype(np.float32)
        return out


class MockDecisionAdapter:
    """Política de aceptación simulada (umbral sobre el máximo softmax).

    Delega lógicamente en "máximo con umbral" para que las pruebas puedan probar
    valores por debajo, iguales y por encima del umbral. En modo real, la
    política y el umbral los entrega Acosta.
    """

    def __init__(
        self, *, threshold: float = 0.55, policy_version: str = MOCK_POLICY_VERSION
    ) -> None:
        self.threshold = float(threshold)
        self.policy_version = policy_version

    def decide(self, probabilities: np.ndarray) -> DecisionResult:
        probs = np.asarray(probabilities, dtype=np.float32)
        if probs.ndim == 2:
            probs = probs[0]
        # Desempate determinista: menor índice gana ante empate exacto.
        label_id = int(np.argmax(probs))
        confidence = float(probs[label_id])
        # Comparación exacta contra el umbral publicado.
        if confidence >= self.threshold:
            return DecisionResult(
                accepted=True,
                label_id=label_id,
                confidence=confidence,
                reason_codes=[],
            )
        return DecisionResult(
            accepted=False,
            label_id=None,
            confidence=None,
            reason_codes=["LOW_CONFIDENCE"],
        )


# Etiquetas simuladas en el ORDEN CANÓNICO. En modo real provienen de
# labels.json (Acosta). Se usan sólo para que /config y la UI tengan textos.
MOCK_LABELS = [
    "Hola",
    "Buenos",
    "Días",
    "Tardes",
    "Noches",
    "Yo",
    "Nombre",
    "Años",
    "Gustar",
    "Licor",
]
