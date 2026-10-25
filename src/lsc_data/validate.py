"""Validación de un bundle de características (design §18/§19).

``validate(bundle_dir) -> int`` comprueba esquema, formas, integridad, cobertura
y fuga del bundle producido por ``packaging.write_bundle``:

- ``0`` conforme,
- ``2`` entradas o configuración inválidas (archivos ausentes, spec mal formado),
- ``3`` fallo de integridad (formas incorrectas, ``y`` fuera de 0..9, duplicados,
  cobertura de clases rota).

Devuelve el código de salida; no imprime al usuario (la CLI traduce/loggea).
"""

from __future__ import annotations

import json
from pathlib import Path

from contracts.canonical import feature_spec_hash as _feature_spec_hash
from contracts.models import NUM_CLASSES

from .errors import ConfigError, IntegrityError
from .packaging import (
    FEATURE_SPEC_JSON,
    FEATURES_NPZ,
    read_features_npz,
)

__all__ = ["validate"]

_EXPECTED_FRAME_SHAPE = (6, 147)


def validate(bundle_dir: str | Path) -> int:
    """Valida el bundle; devuelve el código de salida 0/2/3.

    Lanza ``ConfigError`` (-> 2) para entradas ausentes/mal formadas e
    ``IntegrityError`` (-> 3) para fallos de integridad, que la CLI mapea; si
    todo es conforme, devuelve ``0``.
    """
    import numpy as np

    bundle = Path(bundle_dir)
    npz_path = bundle / FEATURES_NPZ
    spec_path = bundle / FEATURE_SPEC_JSON
    if not npz_path.is_file():
        raise ConfigError(f"bundle sin {FEATURES_NPZ}: {npz_path}")
    if not spec_path.is_file():
        raise ConfigError(f"bundle sin {FEATURE_SPEC_JSON}: {spec_path}")

    # --- feature_spec.json: estructura y hash semántico coherente ---
    try:
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"feature_spec.json mal formado: {exc}") from exc
    stored_hash = spec.get("feature_spec_hash")
    if not isinstance(stored_hash, str):
        raise ConfigError("feature_spec.json: 'feature_spec_hash' ausente")
    recomputed = _feature_spec_hash(spec)
    if recomputed != stored_hash:
        raise IntegrityError(
            f"feature_spec_hash divergente: esperado {recomputed}, "
            f"encontrado {stored_hash}"
        )

    # --- features.npz: tipos, formas e integridad (allow_pickle=False) ---
    data = read_features_npz(npz_path)

    if data.X.dtype != np.float32:
        raise IntegrityError(f"X.dtype debe ser float32; es {data.X.dtype}")
    if data.X.ndim != 3 or data.X.shape[1:] != _EXPECTED_FRAME_SHAPE:
        raise IntegrityError(
            f"X debe tener forma (N, 6, 147); es {data.X.shape}"
        )
    n = data.X.shape[0]

    for name, arr, dtype in (
        ("y", data.y, np.int64),
        ("quality_pass", data.quality_pass, np.bool_),
    ):
        if arr.dtype != dtype:
            raise IntegrityError(f"{name}.dtype debe ser {dtype}; es {arr.dtype}")

    for name, arr in (
        ("y", data.y),
        ("sample_id", data.sample_id),
        ("signer_id", data.signer_id),
        ("split", data.split),
        ("quality_pass", data.quality_pass),
    ):
        if arr.shape[0] != n:
            raise IntegrityError(f"longitud de {name} ({arr.shape[0]}) != N ({n})")

    # y en 0..9 (negativos usan paquete propio; aquí el bundle de 10 clases).
    y_vals = data.y.tolist()
    out_of_range = [v for v in y_vals if not (0 <= int(v) <= NUM_CLASSES - 1)]
    if out_of_range:
        raise IntegrityError(f"y fuera de 0..9: {sorted(set(out_of_range))}")

    # sample_id únicos.
    sample_ids = [str(s) for s in data.sample_id.tolist()]
    if len(set(sample_ids)) != len(sample_ids):
        raise IntegrityError("sample_id duplicados en el bundle")

    # Fuga: un signer_id no aparece en más de un split.
    signer_to_splits: dict[str, set[str]] = {}
    for signer, split in zip(data.signer_id.tolist(), data.split.tolist()):
        signer_to_splits.setdefault(str(signer), set()).add(str(split))
    leaked = {s: sorted(v) for s, v in signer_to_splits.items() if len(v) > 1}
    if leaked:
        raise IntegrityError(f"fuga de identidad entre splits: {leaked}")

    return 0
