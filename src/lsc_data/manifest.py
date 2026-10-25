"""Construcción del manifiesto (design §9, §18).

``build_manifest(raw_dir, source_meta, label_map) -> ManifestReport``: recorre
``raw_dir``, interpreta cada frame con la gramática LSC70W, agrupa en secuencias
de 6 y escribe un manifiesto JSONL más un ``exclusions.csv`` con las muestras
descartadas. Config o ``label_map`` inválido es fatal (``ConfigError`` -> exit
2). Un registro corrupto/ambiguo NO detiene el proceso: va a ``exclusions.csv``
y se continúa (design §18).

Este módulo no lee píxeles: el hash de cada frame (``frame_sha256``) se calcula
sobre los bytes del archivo, y las dimensiones se obtienen de forma perezosa con
OpenCV sólo si está disponible; sin corpus real este camino no se ejercita (los
tests usan ``build_manifest_from_frames`` con entradas sintéticas).
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Iterable, Mapping

from contracts.models import ManifestRecord, ManifestReport

from .adapter_lsc70w import GrammarConfig, Lsc70wAdapter
from .errors import ConfigError
from .sequence import (
    FrameInput,
    SequenceOutcome,
    build_record,
    group_frames,
)

__all__ = [
    "build_manifest_from_frames",
    "build_manifest",
    "write_manifest",
    "write_exclusions",
]

EXCLUSIONS_HEADER = ("artifact_id", "stage", "reason_code", "detail")


def _raw_label_by_class(label_map: Mapping[str, int]) -> dict[int, str]:
    """Deriva un texto de origen por ``class_id`` a partir del label_map.

    Usa la etiqueta de carpeta como ``raw_label`` cuando no hay otra fuente.
    """
    out: dict[int, str] = {}
    for label, class_id in label_map.items():
        out.setdefault(class_id, label)
    return out


def _detect_cross_sequence_copies(
    outcomes: list[SequenceOutcome],
) -> list[tuple[int, str]]:
    """Marca copias cruzadas: mismo ``sequence_sha256`` y distinto ``signer_id``.

    Devuelve ``(índice, detalle)`` por cada muestra válida que comparte su
    ``sequence_sha256`` con otra de un ``signer_id`` distinto (design §6.3/D06).
    """
    by_seq: dict[str, list[int]] = {}
    for i, outcome in enumerate(outcomes):
        if outcome.record is not None:
            by_seq.setdefault(outcome.record.sequence_sha256, []).append(i)
    flagged: list[tuple[int, str]] = []
    for seq_hash, idxs in by_seq.items():
        signers = {outcomes[i].record.signer_id for i in idxs}  # type: ignore[union-attr]
        if len(idxs) > 1 and len(signers) > 1:
            for i in idxs:
                flagged.append(
                    (i, f"copia cruzada: sequence_sha256={seq_hash} en {sorted(signers)}")
                )
    return flagged


def build_manifest_from_frames(
    frames: Iterable[FrameInput],
    *,
    source: str,
    source_version: str,
    label_map: Mapping[str, int],
    manifest_path: str | Path,
    exclusions_path: str | Path,
) -> ManifestReport:
    """Núcleo testeable: agrupa frames ya interpretados y escribe artefactos.

    ``label_map`` vacío es ``ConfigError`` (exit 2). Las muestras inválidas y
    las copias cruzadas van a ``exclusions.csv``; el resto al manifiesto JSONL.
    """
    if not label_map:
        raise ConfigError("build_manifest: label_map vacío o inválido")

    candidates = group_frames(
        frames,
        source=source,
        source_version=source_version,
        raw_label_by_class=_raw_label_by_class(label_map),
    )
    outcomes = [build_record(c) for c in candidates]

    # Copias cruzadas: degradan muestras válidas a exclusión (incidente D06).
    for idx, detail in _detect_cross_sequence_copies(outcomes):
        prev = outcomes[idx]
        outcomes[idx] = SequenceOutcome(
            record=None,
            reason_code="CONFIG_MISMATCH",
            detail=detail,
            artifact_id=prev.artifact_id,
        )

    records: list[ManifestRecord] = []
    exclusions: list[tuple[str, str, str, str]] = []
    by_class: dict[int, int] = {}
    by_signer: dict[str, int] = {}

    for outcome in outcomes:
        if outcome.record is not None:
            rec = outcome.record
            records.append(rec)
            by_class[rec.class_id] = by_class.get(rec.class_id, 0) + 1
            by_signer[rec.signer_id] = by_signer.get(rec.signer_id, 0) + 1
        else:
            exclusions.append(
                (
                    outcome.artifact_id,
                    "manifest",
                    outcome.reason_code or "UNKNOWN",
                    outcome.detail,
                )
            )

    write_manifest(records, manifest_path)
    write_exclusions(exclusions, exclusions_path)

    return ManifestReport(
        n_records=len(outcomes),
        n_valid=len(records),
        n_excluded=len(exclusions),
        by_class={str(k): v for k, v in sorted(by_class.items())},
        by_signer=dict(sorted(by_signer.items())),
        exclusions_path=str(exclusions_path),
        manifest_path=str(manifest_path),
        warnings=[],
    )


def write_manifest(records: Iterable[ManifestRecord], path: str | Path) -> None:
    """Escribe el manifiesto como JSONL ordenado por ``sample_id``."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(records, key=lambda r: r.sample_id)
    with out.open("w", encoding="utf-8", newline="\n") as handle:
        for rec in ordered:
            handle.write(rec.model_dump_json(exclude_none=True))
            handle.write("\n")


def write_exclusions(
    rows: Iterable[tuple[str, str, str, str]], path: str | Path
) -> None:
    """Escribe ``exclusions.csv`` con encabezado fijo (design §4.3)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(EXCLUSIONS_HEADER)
        for row in rows:
            writer.writerow(row)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(
    raw_dir: str | Path,
    source_meta: GrammarConfig,
    label_map: Mapping[str, int] | None = None,
    *,
    manifest_path: str | Path,
    exclusions_path: str | Path,
) -> ManifestReport:
    """Construye el manifiesto recorriendo ``raw_dir`` (camino con corpus real).

    ``source_meta`` es el ``GrammarConfig`` resuelto de ``lsc70w.yaml``; si
    ``label_map`` es ``None`` se usa el del ``GrammarConfig``. Interpreta cada
    archivo bajo ``raw_dir`` con la gramática; las rutas que no casan son
    ``ConfigError`` (fatal, exit 2). Las dimensiones se leen de forma perezosa
    con OpenCV. Este camino no se ejercita sin corpus real (bloque DO07); los
    tests usan ``build_manifest_from_frames``.
    """
    effective_label_map = dict(label_map) if label_map else dict(source_meta.label_map)
    if not effective_label_map:
        raise ConfigError("build_manifest: label_map vacío o inválido")

    root = Path(raw_dir)
    if not root.is_dir():
        raise ConfigError(f"raw_dir no es un directorio: {root}")

    adapter = Lsc70wAdapter(source_meta)
    frames: list[FrameInput] = []
    for file_path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = file_path.relative_to(root).as_posix()
        parsed = adapter.parse_frame(rel)  # ConfigError si no casa
        width, height = _read_image_size(file_path)
        frames.append(
            FrameInput(
                parsed=parsed,
                frame_sha256=_sha256_file(file_path),
                width=width,
                height=height,
            )
        )

    return build_manifest_from_frames(
        frames,
        source=source_meta.source,
        source_version=source_meta.source_version,
        label_map=effective_label_map,
        manifest_path=manifest_path,
        exclusions_path=exclusions_path,
    )


def _read_image_size(path: Path) -> tuple[int, int]:
    """Lee (width, height) con OpenCV (import perezoso; bloque DO07)."""
    try:
        import cv2  # import perezoso: sólo en el camino con corpus real
    except ModuleNotFoundError as exc:  # pragma: no cover - sin corpus real
        raise ConfigError(
            "opencv-python-headless no está instalado; necesario para leer "
            "dimensiones de imagen en build_manifest sobre el corpus real."
        ) from exc
    import numpy as np  # noqa: F401 - cv2 depende de numpy

    image = cv2.imread(str(path))  # pragma: no cover - sin corpus real
    if image is None:  # pragma: no cover - sin corpus real
        raise ConfigError(f"imagen ilegible: {path}")
    height, width = image.shape[:2]  # pragma: no cover - sin corpus real
    return int(width), int(height)  # pragma: no cover - sin corpus real
