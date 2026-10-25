"""Empaquetado tipado sin pickle (design §17).

``features.npz`` contiene arrays TIPADOS, sin object arrays ni pickle:

- ``X``          float32 ``(N, 6, 147)``
- ``y``          int64 ``(N,)``
- ``sample_id``  unicode ``(N,)``
- ``signer_id``  unicode ``(N,)``
- ``split``      unicode ``(N,)``
- ``quality_pass`` bool ``(N,)``

Se escribe con ``np.savez`` y se lee SIEMPRE con ``allow_pickle=False``. El
bundle también incluye ``feature_spec.json`` (§4.2/§6.1) y ``exclusions.csv``.
``numpy`` se importa de forma perezosa para no acoplar el import del paquete.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from contracts.canonical import feature_spec_hash as _feature_spec_hash

from .errors import ConfigError
from .manifest import write_exclusions

__all__ = [
    "FeatureBundle",
    "write_bundle",
    "write_features_npz",
    "read_features_npz",
    "write_feature_spec",
]

FEATURES_NPZ = "features.npz"
FEATURE_SPEC_JSON = "feature_spec.json"
EXCLUSIONS_CSV = "exclusions.csv"

_NPZ_SHAPE = (6, 147)


@dataclass(frozen=True)
class FeatureBundle:
    """Contenido tipado de ``features.npz`` cargado en memoria."""

    X: Any  # float32 (N, 6, 147)
    y: Any  # int64 (N,)
    sample_id: Any  # unicode (N,)
    signer_id: Any  # unicode (N,)
    split: Any  # unicode (N,)
    quality_pass: Any  # bool (N,)


def write_features_npz(
    path: str | Path,
    *,
    X: Any,
    y: Sequence[int],
    sample_id: Sequence[str],
    signer_id: Sequence[str],
    split: Sequence[str],
    quality_pass: Sequence[bool],
) -> Path:
    """Escribe ``features.npz`` con dtypes fijos (design §17).

    Valida formas y dtype de ``X`` antes de guardar: ``(N, 6, 147)`` float32.
    No usa object arrays ni pickle.
    """
    import numpy as np

    arr_x = np.asarray(X, dtype=np.float32)
    n = arr_x.shape[0]
    if arr_x.shape[1:] != _NPZ_SHAPE:
        raise ConfigError(
            f"X debe tener forma (N, 6, 147); tiene {arr_x.shape}"
        )
    arr_y = np.asarray(y, dtype=np.int64)
    arr_sample = np.asarray(sample_id, dtype=np.str_)
    arr_signer = np.asarray(signer_id, dtype=np.str_)
    arr_split = np.asarray(split, dtype=np.str_)
    arr_quality = np.asarray(quality_pass, dtype=np.bool_)

    for name, arr in (
        ("y", arr_y),
        ("sample_id", arr_sample),
        ("signer_id", arr_signer),
        ("split", arr_split),
        ("quality_pass", arr_quality),
    ):
        if arr.shape[0] != n:
            raise ConfigError(
                f"longitud de {name} ({arr.shape[0]}) != N ({n})"
            )

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        X=arr_x,
        y=arr_y,
        sample_id=arr_sample,
        signer_id=arr_signer,
        split=arr_split,
        quality_pass=arr_quality,
    )
    # np.savez añade .npz si falta; normalizamos el retorno.
    if out.suffix != ".npz":
        return out.with_suffix(".npz")
    return out


def read_features_npz(path: str | Path) -> FeatureBundle:
    """Lee ``features.npz`` con ``allow_pickle=False`` (design §17)."""
    import numpy as np

    with np.load(str(path), allow_pickle=False) as data:
        return FeatureBundle(
            X=data["X"],
            y=data["y"],
            sample_id=data["sample_id"],
            signer_id=data["signer_id"],
            split=data["split"],
            quality_pass=data["quality_pass"],
        )


def write_feature_spec(path: str | Path, spec: Mapping[str, Any]) -> Path:
    """Escribe ``feature_spec.json`` con su ``feature_spec_hash`` (§6.1).

    El hash se recomputa desde el propio ``spec`` (excluyendo ``build_date`` y
    campos ``*_hash``), de modo que un rename cosmético o un cambio de fecha no
    lo altera.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    doc = dict(spec)
    doc["feature_spec_hash"] = _feature_spec_hash(spec)
    out.write_text(
        json.dumps(doc, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return out


def write_bundle(
    out_dir: str | Path,
    *,
    X: Any,
    y: Sequence[int],
    sample_id: Sequence[str],
    signer_id: Sequence[str],
    split: Sequence[str],
    quality_pass: Sequence[bool],
    feature_spec: Mapping[str, Any],
    exclusions: Sequence[tuple[str, str, str, str]] = (),
) -> Path:
    """Escribe el bundle completo: ``features.npz`` + spec + exclusiones."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_features_npz(
        out / FEATURES_NPZ,
        X=X,
        y=y,
        sample_id=sample_id,
        signer_id=signer_id,
        split=split,
        quality_pass=quality_pass,
    )
    write_feature_spec(out / FEATURE_SPEC_JSON, feature_spec)
    write_exclusions(list(exclusions), out / EXCLUSIONS_CSV)
    return out
