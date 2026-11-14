"""Predictor real (``model.keras``) y Predictor simulado determinista.

Contrato (plan §8): ``predict(x)`` recibe un lote float32 ``(B, 6, 147)`` y
retorna probabilidades ``(B, 10)`` en el orden de ``labels.json``. No aplica
otra Softmax: el modelo ya termina en softmax. La entrada se valida (forma,
dtype convertible, finitud); una entrada inválida lanza ``PredictorInputError``.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

SEQ_SHAPE = (6, 147)
NUM_CLASSES = 10


class PredictorInputError(ValueError):
    """Entrada con forma o valores incompatibles con el contrato."""


def _validate_input(x) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[None, ...]
    if arr.ndim != 3 or arr.shape[1:] != SEQ_SHAPE:
        raise PredictorInputError(f"forma {arr.shape} incompatible con (B,{SEQ_SHAPE[0]},{SEQ_SHAPE[1]})")
    if arr.shape[0] == 0:
        raise PredictorInputError("lote vacío")
    if not np.all(np.isfinite(arr)):
        raise PredictorInputError("la entrada contiene valores no finitos")
    return arr


class Predictor:
    """Carga ``model.keras`` y expone ``predict``."""

    def __init__(self, model_path: str | Path):
        import keras

        from src.lsc_models import custom_layers  # noqa: F401  (registra capas)

        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(f"no existe el modelo: {self.model_path}")
        self._model = keras.models.load_model(self.model_path, compile=False)
        ishape, oshape = tuple(self._model.input_shape[1:]), tuple(self._model.output_shape[1:])
        if ishape != SEQ_SHAPE or oshape != (NUM_CLASSES,):
            raise ValueError(f"modelo incompatible: entrada {ishape}, salida {oshape}")
        self.model_sha256 = hashlib.sha256(self.model_path.read_bytes()).hexdigest()

    def predict(self, x) -> np.ndarray:
        arr = _validate_input(x)
        probs = self._model(arr, training=False)
        return np.asarray(probs, dtype=np.float32)


class SimulatedPredictor:
    """DOBLE DE PRUEBA determinista. Nunca entra en la demostración ni en métricas.

    Las probabilidades salen de un hash del contenido de ``x``: misma entrada,
    misma salida. ``is_simulated`` permite a la interfaz mostrar el rótulo de
    "modo de prueba".
    """

    is_simulated = True
    model_version = "simulated-predictor-0.0.0"

    def predict(self, x) -> np.ndarray:
        arr = _validate_input(x)
        out = np.empty((arr.shape[0], NUM_CLASSES), dtype=np.float32)
        for i in range(arr.shape[0]):
            seed = int.from_bytes(hashlib.sha256(np.ascontiguousarray(arr[i]).tobytes()).digest()[:8], "big")
            logits = np.random.default_rng(seed).normal(0.0, 2.0, NUM_CLASSES)
            e = np.exp(logits - logits.max())
            out[i] = (e / e.sum()).astype(np.float32)
        return out
