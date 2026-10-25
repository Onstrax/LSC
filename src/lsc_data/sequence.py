"""Reconstrucción de secuencias (design §9).

Agrupa frames por ``(source, signer_id, class_id, repetition_id)``, ordena
numéricamente por ``frame_index`` y valida que cada muestra tenga exactamente 6
índices únicos. Las muestras que no cumplen se excluyen con un ``reason_code``;
NUNCA se completa con frames de otra ejecución ni se replican frames. Un signo
estático puede tener imágenes idénticas (hashes iguales) y eso NO invalida.

``sample_id`` y ``sequence_sha256`` se calculan vía ``contracts.canonical``
(re-exportado en ``hashing``). Este módulo no lee imágenes: recibe, por frame,
una ruta relativa interpretada por el adapter y su hash (``frame_sha256``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from contracts.models import ManifestRecord

from .adapter_lsc70w import ParsedFrame
from .hashing import sample_id as _sample_id
from .hashing import sequence_sha256 as _sequence_sha256

__all__ = [
    "FrameInput",
    "SequenceCandidate",
    "SequenceOutcome",
    "group_frames",
    "build_record",
]

SEQUENCE_LENGTH = 6
MANIFEST_SCHEMA_VERSION = "lsc.manifest.v1"


@dataclass(frozen=True)
class FrameInput:
    """Un frame candidato: su interpretación de gramática más metadatos."""

    parsed: ParsedFrame
    frame_sha256: str
    width: int
    height: int


@dataclass(frozen=True)
class SequenceCandidate:
    """Grupo de frames candidatos a una muestra (clave de agrupamiento)."""

    source: str
    source_version: str
    signer_id: str
    class_id: int
    repetition_id: int
    raw_label: str
    frames: tuple[FrameInput, ...]


@dataclass(frozen=True)
class SequenceOutcome:
    """Resultado de reconstruir una muestra: válida o excluida."""

    record: ManifestRecord | None
    reason_code: str | None = None
    detail: str = ""
    artifact_id: str = ""
    excluded_fields: dict = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        return self.record is not None


def group_frames(
    frames: Iterable[FrameInput],
    *,
    source: str,
    source_version: str,
    raw_label_by_class: dict[int, str],
) -> list[SequenceCandidate]:
    """Agrupa frames por ``(source, signer_id, class_id, repetition_id)``.

    El orden de salida es estable: ordenado por la clave de agrupamiento. El
    ``raw_label`` se toma de ``raw_label_by_class`` (texto de origen por clase).
    """
    groups: dict[tuple[str, int, int], list[FrameInput]] = {}
    for frame in frames:
        key = (
            frame.parsed.signer_id,
            frame.parsed.class_id,
            frame.parsed.repetition_id,
        )
        groups.setdefault(key, []).append(frame)

    candidates: list[SequenceCandidate] = []
    for (signer_id, class_id, repetition_id) in sorted(groups):
        members = groups[(signer_id, class_id, repetition_id)]
        candidates.append(
            SequenceCandidate(
                source=source,
                source_version=source_version,
                signer_id=signer_id,
                class_id=class_id,
                repetition_id=repetition_id,
                raw_label=raw_label_by_class.get(class_id, ""),
                frames=tuple(members),
            )
        )
    return candidates


def _artifact_id(candidate: SequenceCandidate) -> str:
    return (
        f"{candidate.source_version}/{candidate.signer_id}/"
        f"{candidate.class_id}/{candidate.repetition_id}"
    )


def build_record(candidate: SequenceCandidate) -> SequenceOutcome:
    """Valida un grupo y construye una ``ManifestRecord`` o una exclusión.

    Reglas (design §9):
    - exactamente 6 frames con ``frame_index`` ÚNICOS -> válida; si no, exclusión;
    - se ordena numéricamente por ``frame_index`` antes de hashear;
    - imágenes idénticas (hashes repetidos) NO invalidan un signo estático;
    - NUNCA se completan frames faltantes con otra ejecución.
    """
    artifact_id = _artifact_id(candidate)
    frames = candidate.frames

    indices = [f.parsed.frame_index for f in frames]
    if len(set(indices)) != len(indices):
        return SequenceOutcome(
            record=None,
            reason_code="SIZE_MISMATCH",
            detail=f"frame_index duplicado (ambiguo) en {indices}",
            artifact_id=artifact_id,
        )

    if len(frames) != SEQUENCE_LENGTH:
        return SequenceOutcome(
            record=None,
            reason_code="INVALID_FRAME_COUNT",
            detail=f"se esperaban {SEQUENCE_LENGTH} frames, hay {len(frames)}",
            artifact_id=artifact_id,
        )

    ordered = sorted(frames, key=lambda f: f.parsed.frame_index)

    widths = {f.width for f in ordered}
    heights = {f.height for f in ordered}
    if len(widths) != 1 or len(heights) != 1:
        return SequenceOutcome(
            record=None,
            reason_code="SIZE_MISMATCH",
            detail=f"dimensiones mezcladas width={widths} height={heights}",
            artifact_id=artifact_id,
        )

    frame_paths = [f.parsed.frame_path for f in ordered]
    frame_sha256 = [f.frame_sha256 for f in ordered]
    frame_indices = [f.parsed.frame_index for f in ordered]
    width = ordered[0].width
    height = ordered[0].height

    seq_hash = _sequence_sha256(frame_sha256)
    sid = _sample_id(
        source_version=candidate.source_version,
        signer_id=candidate.signer_id,
        class_id=candidate.class_id,
        repetition_id=candidate.repetition_id,
        frame_sha256=frame_sha256,
    )

    record = ManifestRecord(
        schema_version=MANIFEST_SCHEMA_VERSION,
        sample_id=sid,
        source=candidate.source,
        source_version=candidate.source_version,
        signer_id=candidate.signer_id,
        class_id=candidate.class_id,
        raw_label=candidate.raw_label,
        repetition_id=candidate.repetition_id,
        frame_paths=frame_paths,
        frame_sha256=frame_sha256,
        frame_indices=frame_indices,
        width=width,
        height=height,
        n_frames=SEQUENCE_LENGTH,
        sequence_sha256=seq_hash,
        structural_status="valid",
    )
    return SequenceOutcome(record=record, artifact_id=artifact_id)
