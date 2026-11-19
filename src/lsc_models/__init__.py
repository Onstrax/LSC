"""Modelos, experimentos y evaluación del proyecto LSC (módulo de Acosta).

Importar este paquete NO carga TensorFlow: los imports pesados son perezosos
(``architectures``, ``train``) para que los tests de datos/métricas corran sin él.

Rutas propias (AGENTS.md): ``src/lsc_models/``, ``configs/train/``,
``reports/models/``, ``tests/models/``.
"""

from __future__ import annotations

__all__ = ["config", "data", "metrics", "baseline"]
__version__ = "0.1.0"
