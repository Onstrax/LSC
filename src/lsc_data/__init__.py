"""Paquete de datos sin red del módulo de Ortiz (design §3, §9, §10, §17).

API pública estable (firmas que los demás componentes consumen):

- ``build_manifest(raw_dir, source_meta, label_map, *, manifest_path,
  exclusions_path) -> ManifestReport`` y su núcleo testeable
  ``build_manifest_from_frames``.
- ``make_splits(manifest_path, seed=20261022, *, out_path=None) -> SplitReport``.
- ``inventory(raw_dir, out_path) -> list[tuple[str, int, int]]``.
- ``acquire(config_path, *, archive=None, dest_dir=None) -> list[str]``.
- ``safe_extract(archive_path, dest_dir, *, allowed_extensions=None)``.
- ``write_bundle``/``read_features_npz``/``write_features_npz`` (features.npz
  tipado, sin pickle) y ``validate(bundle_dir) -> int``.
- Hashing canónico re-exportado de ``contracts.canonical`` vía ``hashing``.

Importar este paquete NO carga detectores, NO importa cv2/mediapipe/PyYAML/numpy
ni toca disco: esas dependencias se importan de forma perezosa dentro de cada
función que las necesita (design §16).
"""

from __future__ import annotations

from .acquisition import acquire, load_source_config
from .adapter_lsc70w import Lsc70wAdapter, load_grammar_config
from .errors import ConfigError, DataError, IntegrityError
from .hashing import (
    feature_spec_hash,
    manifest_sha256,
    sample_id,
    sequence_sha256,
    sha256_hex,
)
from .inventory import inventory
from .manifest import build_manifest, build_manifest_from_frames
from .packaging import (
    FeatureBundle,
    read_features_npz,
    write_bundle,
    write_feature_spec,
    write_features_npz,
)
from .security import safe_extract
from .split import make_splits, split_counts
from .validate import validate

__all__ = [
    # errores tipados
    "DataError",
    "ConfigError",
    "IntegrityError",
    # hashing (re-export de contracts.canonical)
    "sample_id",
    "sequence_sha256",
    "manifest_sha256",
    "feature_spec_hash",
    "sha256_hex",
    # adquisición y seguridad
    "acquire",
    "load_source_config",
    "safe_extract",
    # gramática del corpus
    "load_grammar_config",
    "Lsc70wAdapter",
    # manifiesto e inventario
    "build_manifest",
    "build_manifest_from_frames",
    "inventory",
    # splits
    "make_splits",
    "split_counts",
    # empaquetado y validación
    "FeatureBundle",
    "write_bundle",
    "write_features_npz",
    "read_features_npz",
    "write_feature_spec",
    "validate",
]
