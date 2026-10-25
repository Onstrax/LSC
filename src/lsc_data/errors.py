"""Excepciones tipadas internas del módulo de datos (design §18).

Estas excepciones no se muestran al usuario final: la CLI (``cli.py``) las
mapea a los códigos de salida uniformes ``0/2/3`` (design §19). El mapeo es:

- ``ConfigError``   -> exit 2 (entrada o configuración inválida).
- ``IntegrityError`` -> exit 3 (fallo de integridad: fuga, hash divergente,
  rutas peligrosas en extracción, formas o cobertura rotas).

``DataError`` es la base común para poder capturar cualquier error del módulo.
"""

from __future__ import annotations

__all__ = [
    "DataError",
    "ConfigError",
    "IntegrityError",
]


class DataError(Exception):
    """Base de todos los errores tipados del módulo de datos."""


class ConfigError(DataError):
    """Entrada o configuración inválida (YAML mal formado, regex que no casa,

    esquema de config incompatible, ``label_map`` inválido). Mapea a exit 2.
    """


class IntegrityError(DataError):
    """Fallo de integridad (design §18/§19). Mapea a exit 3.

    Cubre fuga de identidad/muestras entre splits, duplicados por
    ``sequence_sha256``, rutas peligrosas detectadas en la extracción, hash
    divergente en ``--resume`` y formas/cobertura rotas en ``validate``.
    """
