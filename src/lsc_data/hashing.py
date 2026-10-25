"""Hashing de datos — re-export de ``contracts.canonical`` (design §6).

Fuente ÚNICA de hashing del módulo de datos. No se reimplementa nada aquí: se
reexportan ``sample_id``, ``sequence_sha256``, ``manifest_sha256``,
``feature_spec_hash``, ``sha256_hex`` y ``canonical_bytes`` de FEAT-001 para
que ``src/lsc_data`` tenga un único punto de acceso sin duplicar la regla
canónica.
"""

from __future__ import annotations

from contracts.canonical import (
    canonical_bytes,
    feature_spec_hash,
    manifest_sha256,
    sample_id,
    sequence_sha256,
    sha256_hex,
)

__all__ = [
    "canonical_bytes",
    "sha256_hex",
    "sample_id",
    "sequence_sha256",
    "manifest_sha256",
    "feature_spec_hash",
]
