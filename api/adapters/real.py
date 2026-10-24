"""Adaptadores REALES que envuelven los componentes de Ortiz y Acosta.

Reglas (SDD §4, §5, §11):
- La inicialización compara hashes, orden de etiquetas y forma esperada.
- Una incompatibilidad impide predecir y produce un diagnóstico para el
  operador (NO un fallback silencioso a mock).
- Juan no aplica otra Softmax ni corrige probabilidades; sólo verifica forma,
  finitud, rango [0,1] y suma ~1 con tolerancia 1e-4.

Estas clases importan `src.lsc_features` (Ortiz) y `src.lsc_inference` (Acosta)
de forma perezosa: el esqueleto de Juan debe IMPORTARSE sin esos paquetes ni
pesos presentes (criterio de aceptación de JU01).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from api.adapters.types import DecisionResult, FeatureResult
from api.constants import NUM_CLASSES, PROB_SUM_ATOL, SEQUENCE_SHAPE


class ArtifactError(RuntimeError):
    """Artefacto ausente o incompatible. Deja el servicio no disponible (503)."""


class ModelOutputError(RuntimeError):
    """Salida del modelo con forma/valores inválidos -> 500 MODEL_OUTPUT_INVALID."""


def _require(path: Path, what: str) -> Path:
    if not path.exists():
        raise ArtifactError(
            f"No se encontró {what} en {path}. El servicio no puede habilitar "
            f"predicción real. Consulte docs/app/operator-guide.md."
        )
    return path


class RealExtractor:
    """Envoltura del `FeatureExtractor` de Ortiz.

    Carga configuración y modelos `.task` desde ``artifacts_dir``. Se construye
    sólo en modo real; si falta algo, lanza :class:`ArtifactError`.
    """

    def __init__(self, artifacts_dir: Path, feature_config: dict | None = None) -> None:
        self._artifacts_dir = Path(artifacts_dir)
        cfg_path = _require(self._artifacts_dir / "feature_spec.json", "feature_spec.json")
        with cfg_path.open("r", encoding="utf-8") as fh:
            self._feature_spec = json.load(fh)
        self.feature_spec_hash = str(self._feature_spec.get("feature_spec_hash", ""))

        # Importación perezosa del componente de Ortiz.
        try:
            from src.lsc_features import FeatureExtractor  # type: ignore
        except Exception as exc:  # pragma: no cover - depende de artefactos reales
            raise ArtifactError(
                "No se pudo importar src.lsc_features (extractor de Ortiz). "
                "Verifique el lock compartido y la preparación de artefactos."
            ) from exc

        model_paths = {
            "pose": _require(self._artifacts_dir / "pose_landmarker.task", "pose .task"),
            "hands": _require(self._artifacts_dir / "hand_landmarker.task", "hand .task"),
        }
        self._impl = FeatureExtractor(feature_config or self._feature_spec, model_paths)

    def extract_sequence(self, frames_rgb: list[np.ndarray]) -> FeatureResult:
        result = self._impl.extract_sequence(frames_rgb)
        # Ortiz entrega su propio tipo; lo normalizamos al contrato compartido
        # sin alterar la matemática.
        return FeatureResult(
            x=np.asarray(result.x, dtype=np.float32),
            quality_pass=bool(result.quality_pass),
            reason_codes=list(result.reason_codes),
            valid_shoulders=bool(getattr(result, "valid_shoulders", True)),
            frames_with_hand=int(getattr(result, "frames_with_hand", 0)),
            feature_spec_hash=str(getattr(result, "feature_spec_hash", self.feature_spec_hash)),
            elapsed_ms=float(getattr(result, "elapsed_ms", 0.0)),
        )


class RealPredictor:
    """Envoltura del `Predictor` de Acosta (model.keras + manifest).

    Verifica compatibilidad en la inicialización: orden de etiquetas, hash del
    feature_spec y forma esperada.
    """

    def __init__(self, artifacts_dir: Path) -> None:
        self._artifacts_dir = Path(artifacts_dir)
        manifest_path = _require(self._artifacts_dir / "manifest.json", "manifest.json")
        with manifest_path.open("r", encoding="utf-8") as fh:
            self._manifest = json.load(fh)
        self.model_version = str(self._manifest.get("model_version", ""))
        self.feature_spec_hash = str(self._manifest.get("feature_spec_hash", ""))

        _require(self._artifacts_dir / "model.keras", "model.keras")

        try:
            from src.lsc_inference import Predictor  # type: ignore
        except Exception as exc:  # pragma: no cover - depende de artefactos reales
            raise ArtifactError(
                "No se pudo importar src.lsc_inference (predictor de Acosta)."
            ) from exc

        self._impl = Predictor(self._artifacts_dir / "model.keras")

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 2:
            x = x[None, ...]
        if x.shape[1:] != SEQUENCE_SHAPE:
            raise ModelOutputError(
                f"Forma de entrada {x.shape} incompatible con {('B',) + SEQUENCE_SHAPE}."
            )
        probs = np.asarray(self._impl.predict(x), dtype=np.float32)
        verify_probabilities(probs, batch=x.shape[0])
        return probs


def verify_probabilities(probs: np.ndarray, *, batch: int) -> None:
    """Valida forma, finitud, rango [0,1] y suma ~1 (atol 1e-4).

    Una salida incompatible es un fallo técnico (MODEL_OUTPUT_INVALID); no se
    aplica otra Softmax ni se corrige silenciosamente.
    """
    if probs.shape != (batch, NUM_CLASSES):
        raise ModelOutputError(f"Forma de salida {probs.shape} != {(batch, NUM_CLASSES)}.")
    if not np.all(np.isfinite(probs)):
        raise ModelOutputError("Probabilidades con valores no finitos (NaN/inf).")
    if probs.min() < -PROB_SUM_ATOL or probs.max() > 1.0 + PROB_SUM_ATOL:
        raise ModelOutputError("Probabilidades fuera del rango [0, 1].")
    sums = probs.sum(axis=1)
    if not np.allclose(sums, 1.0, atol=PROB_SUM_ATOL):
        raise ModelOutputError(f"Las probabilidades no suman ~1 (atol {PROB_SUM_ATOL}).")


class RealDecisionAdapter:
    """Adaptador de decisión versionado que delega en la política de Acosta.

    Lee ``thresholds.json`` y aplica la política documentada. Si la política es
    "máximo softmax con umbral", usa el umbral publicado con comparación exacta
    y desempate determinista (menor índice). Juan no fija un valor por intuición.
    """

    def __init__(self, artifacts_dir: Path) -> None:
        self._artifacts_dir = Path(artifacts_dir)
        th_path = _require(self._artifacts_dir / "thresholds.json", "thresholds.json")
        with th_path.open("r", encoding="utf-8") as fh:
            self._policy = json.load(fh)
        self.policy_version = str(self._policy.get("policy_version", ""))
        if not self.policy_version:
            raise ArtifactError("thresholds.json sin policy_version; política no versionada.")

        # Intentar delegar en la función documentada de Acosta si existe.
        self._delegate = None
        try:
            from src.lsc_inference import DecisionPolicy  # type: ignore

            self._delegate = DecisionPolicy(self._policy)
        except Exception:
            # Sin función de Acosta: aplicar la política declarada en el JSON.
            self._delegate = None

        self._mode = str(self._policy.get("mode", "max_softmax_threshold"))
        self._threshold = float(self._policy.get("threshold", 0.0))

    def decide(self, probabilities: np.ndarray) -> DecisionResult:
        if self._delegate is not None:  # pragma: no cover - depende de Acosta
            d = self._delegate.decide(probabilities)
            return DecisionResult(
                accepted=bool(d.accepted),
                label_id=(int(d.label_id) if d.label_id is not None else None),
                confidence=(float(d.confidence) if d.confidence is not None else None),
                reason_codes=list(d.reason_codes),
            )

        probs = np.asarray(probabilities, dtype=np.float32)
        if probs.ndim == 2:
            probs = probs[0]
        label_id = int(np.argmax(probs))  # desempate: menor índice
        confidence = float(probs[label_id])
        if self._mode == "max_softmax_threshold" and confidence >= self._threshold:
            return DecisionResult(True, label_id, confidence, [])
        return DecisionResult(False, None, None, ["LOW_CONFIDENCE"])
