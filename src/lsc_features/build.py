"""Construcción de características sobre un manifiesto (design §16/§17).

``build_features(manifest_path, splits_path, extractor, out_dir)`` procesa las
muestras EN ORDEN de ``sample_id``: un fallo de UNA muestra se registra y no
elimina a las demás; una corrupción de config o un esquema incompatible DETIENE
el proceso. Devuelve un :class:`contracts.models.FeatureBuildReport`.

La decodificación de imágenes (``cv2``) NO está en alcance de este lote y NO se
importa en el import del módulo. El cargador de cuadros es INYECTABLE
(``frame_loader``); el cargador por defecto usa ``cv2`` de forma PEREZOSA (sólo
cuando se invoca con corpus real, bloque DO07). Las pruebas inyectan un cargador
falso que devuelve 6 arreglos RGB uint8.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from contracts.models import FeatureBuildReport

from .extractor import FeatureExtractor

FrameLoader = Callable[[Mapping[str, Any]], "list[np.ndarray]"]


class SchemaError(ValueError):
    """Config corrupta o esquema incompatible: detiene el proceso (design §16)."""


def _default_frame_loader(record: Mapping[str, Any]) -> list[np.ndarray]:  # pragma: no cover - DO07
    """Carga los 6 cuadros RGB uint8 desde disco con ``cv2`` (import perezoso).

    Sólo se usa en el camino real (bloque DO07); nunca en el import del módulo.
    """
    import cv2  # import perezoso: nunca en el import del paquete

    frames: list[np.ndarray] = []
    for rel_path in record["frame_paths"]:
        bgr = cv2.imread(str(rel_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"No se pudo leer el cuadro {rel_path}")
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    return frames


def _read_manifest(manifest_path: Path) -> list[dict]:
    records: list[dict] = []
    with manifest_path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SchemaError(
                    f"Manifiesto corrupto en {manifest_path}:{line_no}: {exc}"
                ) from exc
            if "sample_id" not in record:
                raise SchemaError(
                    f"Registro sin 'sample_id' en {manifest_path}:{line_no}."
                )
            records.append(record)
    return records


def _read_splits(splits_path: Path) -> dict:
    try:
        with splits_path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        raise SchemaError(f"splits.json ilegible en {splits_path}: {exc}") from exc


def build_features(
    manifest_path: str | Path,
    splits_path: str | Path,
    extractor: FeatureExtractor,
    out_dir: str | Path,
    *,
    frame_loader: FrameLoader | None = None,
) -> FeatureBuildReport:
    """Construye características de todas las muestras del manifiesto (design §16).

    Procesa en orden de ``sample_id``. Un fallo de muestra se registra en
    ``failed_sample_ids`` y el proceso continúa. Una corrupción de config/esquema
    lanza :class:`SchemaError` y detiene el proceso.
    """
    manifest_path = Path(manifest_path)
    splits_path = Path(splits_path)
    out_dir = Path(out_dir)
    loader = frame_loader or _default_frame_loader

    records = _read_manifest(manifest_path)
    _read_splits(splits_path)  # valida esquema; detiene si es incompatible
    records.sort(key=lambda rec: rec["sample_id"])

    # Mapa sample_id -> split (desde splits.json), opcional.
    out_dir.mkdir(parents=True, exist_ok=True)

    n_total = len(records)
    n_quality_pass = 0
    failed_sample_ids: list[str] = []
    per_reason_counts: Counter[str] = Counter()

    xs: list[np.ndarray] = []
    sample_ids: list[str] = []
    signer_ids: list[str] = []
    labels: list[int] = []
    quality_flags: list[bool] = []

    for record in records:
        sample_id = str(record["sample_id"])
        try:
            frames = loader(record)
            result = extractor.extract_sequence(frames)
        except Exception:
            failed_sample_ids.append(sample_id)
            continue

        for code in result.reason_codes:
            per_reason_counts[code] += 1
        if result.quality_pass:
            n_quality_pass += 1

        xs.append(np.asarray(result.x, dtype=np.float32))
        sample_ids.append(sample_id)
        signer_ids.append(str(record.get("signer_id", "")))
        class_id = record.get("class_id")
        labels.append(int(class_id) if class_id is not None else -1)
        quality_flags.append(bool(result.quality_pass))

    bundle_path = out_dir / "features.npz"
    if xs:
        np.savez(
            bundle_path,
            X=np.stack(xs).astype(np.float32),
            y=np.asarray(labels, dtype=np.int64),
            sample_id=np.asarray(sample_ids, dtype=np.str_),
            signer_id=np.asarray(signer_ids, dtype=np.str_),
            quality_pass=np.asarray(quality_flags, dtype=bool),
        )

    return FeatureBuildReport(
        n_total=n_total,
        n_quality_pass=n_quality_pass,
        n_failed=len(failed_sample_ids),
        failed_sample_ids=failed_sample_ids,
        feature_spec_hash=extractor.feature_spec_hash,
        bundle_path=str(bundle_path),
        per_reason_counts=dict(per_reason_counts),
    )
