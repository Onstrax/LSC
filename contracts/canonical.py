"""Serialización canónica y SHA-256 — fuente ÚNICA de hashing del módulo.

Regla byte a byte, reproducible (design §6):

    json.dumps(obj, ensure_ascii=False, sort_keys=True,
               separators=(",", ":"), allow_nan=False).encode("utf-8")

Antes de hashear se eliminan los campos NO semánticos: ``build_date``,
``elapsed_ms``, rutas absolutas y cualquier campo que termine en ``_hash``
(p. ej. ``feature_spec_hash``). Ningún identificador o hash de este módulo
depende de ``float``: se usan enteros, strings de hash e índices enteros.
"""

from __future__ import annotations

import hashlib
from typing import Any, Mapping, Sequence

__all__ = [
    "canonical_bytes",
    "sha256_hex",
    "feature_spec_hash",
    "sample_id",
    "sequence_sha256",
    "manifest_sha256",
]

# Claves NO semánticas que se excluyen del hash semántico (design §6).
_NON_SEMANTIC_KEYS = frozenset({"build_date", "elapsed_ms"})


def canonical_bytes(obj: Any) -> bytes:
    """Representación canónica en bytes UTF-8 (design §6).

    ``sort_keys`` ordena claves, ``separators`` elimina espacios,
    ``ensure_ascii=False`` preserva tildes y ``allow_nan=False`` prohíbe
    ``NaN``/``Infinity``.
    """
    import json

    return json.dumps(
        obj,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_hex(obj: Any) -> str:
    """SHA-256 hex de la representación canónica de ``obj`` (design §6)."""
    return hashlib.sha256(canonical_bytes(obj)).hexdigest()


def _strip_non_semantic(obj: Any) -> Any:
    """Elimina recursivamente claves no semánticas y campos ``*_hash``.

    Se aplica sólo a mappings (incluye los anidados dentro de listas); los
    escalares se devuelven tal cual.
    """
    if isinstance(obj, Mapping):
        return {
            key: _strip_non_semantic(value)
            for key, value in obj.items()
            if key not in _NON_SEMANTIC_KEYS and not str(key).endswith("_hash")
        }
    if isinstance(obj, (list, tuple)):
        return [_strip_non_semantic(item) for item in obj]
    return obj


def feature_spec_hash(spec: Mapping[str, Any]) -> str:
    """SHA-256 del ``FeatureSpec`` canónico (design §6.1).

    Excluye ``build_date`` y cualquier campo ``*_hash`` (incluido el propio
    ``feature_spec_hash``) antes de hashear. Un rename cosmético o un cambio de
    fecha NO altera el hash; un cambio en ``point_order`` sí.
    """
    return sha256_hex(_strip_non_semantic(spec))


def sample_id(
    *,
    source_version: str,
    signer_id: str,
    class_id: int | None,
    repetition_id: int,
    frame_sha256: Sequence[str],
) -> str:
    """SHA-256 del objeto canónico de identidad de la muestra (design §6.2).

    Objeto canónico::

        {"class_id":C,"frame_sha256":[...6...],"repetition_id":R,
         "signer_id":"...","source_version":"..."}

    La lista de hashes de frame se toma en el orden recibido (ya ordenado
    temporalmente por el reconstructor de secuencias).
    """
    obj = {
        "class_id": class_id,
        "frame_sha256": list(frame_sha256),
        "repetition_id": repetition_id,
        "signer_id": signer_id,
        "source_version": source_version,
    }
    return sha256_hex(obj)


def sequence_sha256(frame_sha256_list: Sequence[str]) -> str:
    """SHA-256 SOLO sobre la lista de hashes de imagen (design §6.3).

    Sin identidad ni etiqueta. Dos muestras con igual ``sequence_sha256`` y
    distinto ``signer_id`` indican una copia cruzada (incidente/exclusión).
    """
    return sha256_hex(list(frame_sha256_list))


def manifest_sha256(records: Sequence[Mapping[str, Any]]) -> str:
    """SHA-256 de la concatenación canónica ordenada por ``sample_id`` (§6.4).

    Cada registro se serializa canónicamente (sin campos no semánticos) y se
    concatenan en orden de ``sample_id``; el conjunto se hashea una vez. Lo
    consumen ``splits.json`` y ``validate``.
    """
    ordered = sorted(records, key=lambda rec: rec["sample_id"])
    stripped = [_strip_non_semantic(rec) for rec in ordered]
    joined = b"".join(canonical_bytes(rec) for rec in stripped)
    return hashlib.sha256(joined).hexdigest()
