"""Adaptadores intercambiables de extractor, predictor y decisión.

La API depende de *protocolos* (interfaces), no de implementaciones concretas.
Esto permite trabajo independiente con dobles de prueba (mock) y sustituirlos
por los componentes reales de Ortiz y Acosta sin cambiar la interfaz ni la API.

Selección por modo:
- `LSC_MODE=mock`  -> dobles deterministas (sólo en desarrollo).
- `LSC_MODE=real`  -> adaptadores reales; si faltan artefactos, el servicio
  permanece no disponible (nunca hay fallback silencioso a mock).
"""

from api.adapters.types import (
    DecisionResult,
    ExtractorProtocol,
    FeatureResult,
    PredictorProtocol,
    DecisionProtocol,
)

__all__ = [
    "FeatureResult",
    "DecisionResult",
    "ExtractorProtocol",
    "PredictorProtocol",
    "DecisionProtocol",
]
