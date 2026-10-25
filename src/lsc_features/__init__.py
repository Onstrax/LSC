"""Extractor de características de LSC: matemática pura y detector inyectable.

Importar este paquete NO carga ``cv2`` ni ``mediapipe`` (design §8.2/§16): los
imports pesados son perezosos y viven sólo en el camino real NO ejercitado en
este lote (``RealDetector`` en :mod:`detector`, cargador por defecto en
:mod:`build`). La matemática (geometría, asociación, calidad, temporal, augment)
es pura y se prueba con ``FakeDetector`` y fixtures sintéticos.

API pública:
    - :class:`FeatureExtractor` — extractor inyectable (firma fija §8.3).
    - :class:`FeatureResult` — tipo de salida (idéntico al contrato de la API).
    - :func:`build_features` — construcción sobre un manifiesto (§16).
    - :func:`augment_train` — augmentation de entrenamiento ``train_v1`` (§14).
"""

from __future__ import annotations

from .augment import augment_train
from .build import build_features
from .extractor import FeatureExtractor
from .types import FeatureResult

__all__ = [
    "FeatureExtractor",
    "FeatureResult",
    "build_features",
    "augment_train",
]
