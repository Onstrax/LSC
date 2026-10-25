"""Reglas de calidad y veredicto quality_pass (design §13).

``quality_pass`` requiere: 6 cuadros decodificables, a lo sumo 1 persona por
cuadro, hombros válidos en >=5 cuadros y >=1 mano asociada en >=4 cuadros. Al
fallar, ``x`` conserva forma con ceros pero ``quality_pass=False``. Los
``reason_codes`` se emiten en el orden determinista de §13.
``HAND_ASSOCIATION_AMBIGUOUS`` es advertencia si los mínimos se cumplen.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import reason_codes as rc


@dataclass
class QualityCounters:
    """Contadores acumulados durante ``extract_sequence`` (design §13)."""

    frame_count: int = 0
    decodable_frames: int = 0
    valid_shoulder_frames: int = 0
    frames_with_hand: int = 0
    multiple_people: bool = False
    geometry_outlier: bool = False
    hand_association_ambiguous: bool = False
    config_mismatch: bool = False
    invalid_frame_count: bool = False
    invalid_image: bool = False
    size_mismatch: bool = False
    extra_codes: set = field(default_factory=set)


def evaluate_quality(
    counters: QualityCounters,
    *,
    min_valid_shoulder_frames: int,
    min_hand_frames: int,
) -> tuple[bool, list[str]]:
    """Devuelve ``(quality_pass, reason_codes)`` en orden de emisión (§13).

    Códigos estructurales (conteo de cuadros/imagen/tamaño/múltiples personas),
    mínimos incumplidos (pose/manos) y ``GEOMETRY_OUTLIER`` / ``CONFIG_MISMATCH``
    fuerzan ``quality_pass=False``. ``HAND_ASSOCIATION_AMBIGUOUS`` es sólo
    advertencia si los mínimos se cumplen.
    """
    codes: set[str] = set(counters.extra_codes)
    blocking = False

    if counters.invalid_frame_count:
        codes.add(rc.INVALID_FRAME_COUNT)
        blocking = True
    if counters.invalid_image:
        codes.add(rc.INVALID_IMAGE)
        blocking = True
    if counters.size_mismatch:
        codes.add(rc.SIZE_MISMATCH)
        blocking = True
    if counters.multiple_people:
        codes.add(rc.MULTIPLE_PEOPLE)
        blocking = True
    if counters.config_mismatch:
        codes.add(rc.CONFIG_MISMATCH)
        blocking = True
    if counters.geometry_outlier:
        codes.add(rc.GEOMETRY_OUTLIER)
        blocking = True

    # Mínimos de pose y manos (sólo relevantes si la estructura es válida).
    if counters.valid_shoulder_frames < min_valid_shoulder_frames:
        codes.add(rc.INSUFFICIENT_POSE)
        blocking = True
    if counters.frames_with_hand < min_hand_frames:
        codes.add(rc.INSUFFICIENT_HANDS)
        blocking = True

    if counters.hand_association_ambiguous:
        codes.add(rc.HAND_ASSOCIATION_AMBIGUOUS)

    quality_pass = not blocking
    return quality_pass, rc.order_reason_codes(codes)
