"""Tipo de salida del extractor (design §8.1).

``FeatureResult`` es estructuralmente idéntico al dataclass que la API ya
consume (``api/adapters/types.py``), con una sola diferencia documentada:
``valid_shoulders`` es ``int`` (conteo de cuadros con hombros válidos). La API
lo coacciona por verdad/falsedad (``bool(count)``), de modo que el contrato es
compatible sin tocar ``api/`` (bloque de integración DO08).

**Invariante de salida (design §8.1, hallazgo 4).** Antes de devolver cualquier
``FeatureResult`` (incluso con ``quality_pass=False``), ``x`` cumple
``x.shape == (6, 147)`` y ``x.dtype == np.float32``. El extractor construye ``x``
con ``np.zeros((6, 147), dtype=np.float32)`` y sólo escribe triples válidos; esta
regla es de Ortiz, no de la API.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class FeatureResult:
    """Resultado de ``FeatureExtractor.extract_sequence(frames_rgb)``.

    Atributos tal como los define Ortiz. La API respeta ``quality_pass``: si es
    falso, NO invoca el clasificador.
    """

    x: np.ndarray  # float32 (6, 147)
    quality_pass: bool
    reason_codes: list[str] = field(default_factory=list)
    valid_shoulders: int = 0
    frames_with_hand: int = 0
    feature_spec_hash: str = ""
    elapsed_ms: float = 0.0
