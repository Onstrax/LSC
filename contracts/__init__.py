"""Paquete de contratos del módulo de datos y extracción de Ortiz.

Fuente única de:
- hashing canónico (``contracts.canonical``),
- tipos pydantic v2 del contrato (``contracts.models``),
- artefactos ``labels.json`` y ``feature_spec.json`` en orden canónico,
- JSON Schema Draft 2020-12 (``contracts/schemas``),
- fixtures sintéticos rotulados (``contracts/fixtures``).

Importar este paquete NO carga detectores, no descarga nada ni toca disco.
"""

from __future__ import annotations

__all__ = ["canonical", "models"]
