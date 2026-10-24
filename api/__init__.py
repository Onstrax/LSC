"""Aplicación FastAPI del módulo de Juan Esteban (LSC).

Expone la interfaz web y el contrato HTTP `lsc.http.v1` que coordina el
extractor de características (Ortiz) y el predictor + política de aceptación
(Acosta) mediante adaptadores intercambiables.
"""

__all__ = ["__version__"]

# Versión de la API (api_version en /health). Independiente de la versión del
# modelo y del extractor, que provienen de sus artefactos.
__version__ = "0.1.0"
