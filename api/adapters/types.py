"""Tipos de contrato entre la API y los componentes vecinos.

Estos tipos reflejan LITERALMENTE lo que Ortiz y Acosta entregan (sección 4 del
SDD). La API no reinterpreta la matemática: sólo transporta, valida forma y
coordina. Los protocolos permiten inyectar implementaciones reales o dobles.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np


# ---------------------------------------------------------------------------
# Extractor (Ortiz)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FeatureResult:
    """Resultado de `FeatureExtractor.extract_sequence(frames_rgb)`.

    Atributos tal como los define Ortiz. La API respeta `quality_pass`: si es
    falso, NO invoca el clasificador.
    """

    x: np.ndarray  # float32 (6, 147)
    quality_pass: bool
    reason_codes: list[str] = field(default_factory=list)
    valid_shoulders: bool = True
    frames_with_hand: int = 0
    feature_spec_hash: str = ""
    elapsed_ms: float = 0.0


@runtime_checkable
class ExtractorProtocol(Protocol):
    """Interfaz del extractor de características (propiedad de Ortiz).

    Instanciación real: ``FeatureExtractor(config, model_paths)``.
    """

    #: Hash del feature_spec cargado (para comparar con el del modelo).
    feature_spec_hash: str

    def extract_sequence(self, frames_rgb: list[np.ndarray]) -> FeatureResult:
        """Recibe 6 matrices RGB uint8 del mismo tamaño, orden temporal, sin
        espejo. Devuelve un :class:`FeatureResult`."""
        ...


# ---------------------------------------------------------------------------
# Predictor (Acosta)
# ---------------------------------------------------------------------------
@runtime_checkable
class PredictorProtocol(Protocol):
    """Interfaz del predictor (propiedad de Acosta).

    ``predict(x)`` recibe float32 (B, 6, 147) y devuelve probabilidades
    (B, 10) en [0, 1] con suma próxima a uno. NO devuelve un diccionario: la
    política de aceptación vive en el :class:`DecisionProtocol`.
    """

    #: Versión del modelo (de manifest.json).
    model_version: str
    #: Hash del feature_spec esperado por el modelo.
    feature_spec_hash: str

    def predict(self, x: np.ndarray) -> np.ndarray:
        """float32 (B, 6, 147) -> float32 (B, 10)."""
        ...


# ---------------------------------------------------------------------------
# Decisión (Acosta, adaptada por Juan)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DecisionResult:
    """Salida de ``DecisionAdapter.decide(probabilities)``.

    `accepted` NO significa que una persona haya confirmado la corrección, sólo
    que el sistema ofrece una sugerencia según la política de Acosta.
    """

    accepted: bool
    label_id: int | None
    confidence: float | None
    reason_codes: list[str] = field(default_factory=list)


@runtime_checkable
class DecisionProtocol(Protocol):
    """Interfaz de la política de aceptación (propiedad de Acosta).

    Juan solicita un adaptador versionado que reciba probabilidades y devuelva
    aceptación, clase y razones. El adaptador delega en la función o política
    documentada de Acosta; no fija un umbral por intuición.
    """

    #: Identificador de la versión de la política (de thresholds.json).
    policy_version: str

    def decide(self, probabilities: np.ndarray) -> DecisionResult:
        """probabilities float32 (10,) o (1, 10) -> :class:`DecisionResult`."""
        ...
