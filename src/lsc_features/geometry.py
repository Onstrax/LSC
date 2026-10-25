"""Geometría pura: validez de puntos, centro/escala, normalización (design §11-§12).

Toda la matemática trabaja en píxeles: ``x,y`` normalizados 0..1 se convierten a
px con ``x*W``, ``y*H`` ANTES de distancias o rotaciones. El centro y la escala
de la secuencia son medianas COMUNES a los 6 cuadros. La salida son 49 triples
``(x, y, m)`` por instante, con ``m`` máscara binaria de presencia-del-punto; si
``m=0`` entonces ``x=y=0``. NO se interpola, refleja ni recorta.
"""

from __future__ import annotations

import math

import numpy as np

# Índices de pose conservados (orden del feature_spec): 0, 11..16.
POSE_INDICES = (0, 11, 12, 13, 14, 15, 16)
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_WRIST = 15
RIGHT_WRIST = 16
POINTS_PER_INSTANT = 49  # 7 pose + 21 mano izq + 21 mano der


def is_finite_xy(x: float, y: float) -> bool:
    """``True`` si ambas coordenadas son finitas."""
    return math.isfinite(x) and math.isfinite(y)


def pose_point_valid(
    landmark,
    *,
    visibility_threshold: float = 0.5,
    presence_threshold: float = 0.5,
    use_presence: bool = False,
) -> bool:
    """Regla ``pose_point_valid`` (design §11).

    ``isfinite(x) and isfinite(y) and visibility >= vt and
    (presence >= pt if presence_present else True)``.

    Con ``use_presence=false`` (defecto): la ausencia del campo ``presence`` no
    invalida; su presencia igualmente exige ``presence >= pt``. Con
    ``use_presence=true`` el campo pasa a ser OBLIGATORIO (más estricto; no se
    usa en este lote).
    """
    if landmark is None:
        return False
    x = float(getattr(landmark, "x"))
    y = float(getattr(landmark, "y"))
    if not is_finite_xy(x, y):
        return False
    visibility = float(getattr(landmark, "visibility", 1.0))
    if not (visibility >= visibility_threshold):
        return False
    presence = getattr(landmark, "presence", None)
    if use_presence:
        if presence is None:
            return False
        return float(presence) >= presence_threshold
    if presence is not None:
        return float(presence) >= presence_threshold
    return True


def to_px(x_norm: float, y_norm: float, width: int, height: int) -> tuple[float, float]:
    """Convierte coordenadas normalizadas 0..1 a píxeles (design §12.1)."""
    return x_norm * width, y_norm * height


def shoulder_midpoint_and_distance(
    pose_landmarks: list,
    width: int,
    height: int,
    *,
    min_shoulder_distance_frac_width: float,
    visibility_threshold: float,
    presence_threshold: float,
    use_presence: bool,
) -> tuple[tuple[float, float], float] | None:
    """Punto medio y distancia de hombros en px de un cuadro (design §12.2).

    Devuelve ``((mx, my), dist)`` si los hombros (pose 11 y 12) son válidos y
    ``dist >= min_frac * W``; en caso contrario ``None`` (hombros inválidos para
    ese cuadro). ``pose_landmarks`` es la lista de landmarks de UNA persona.
    """
    if pose_landmarks is None:
        return None
    if len(pose_landmarks) <= max(LEFT_SHOULDER, RIGHT_SHOULDER):
        return None
    left = pose_landmarks[LEFT_SHOULDER]
    right = pose_landmarks[RIGHT_SHOULDER]
    valid_kwargs = dict(
        visibility_threshold=visibility_threshold,
        presence_threshold=presence_threshold,
        use_presence=use_presence,
    )
    if not pose_point_valid(left, **valid_kwargs):
        return None
    if not pose_point_valid(right, **valid_kwargs):
        return None
    lx, ly = to_px(float(left.x), float(left.y), width, height)
    rx, ry = to_px(float(right.x), float(right.y), width, height)
    dist = math.hypot(lx - rx, ly - ry)
    if dist < min_shoulder_distance_frac_width * width:
        return None
    return ((lx + rx) / 2.0, (ly + ry) / 2.0), dist


def sequence_center_scale(
    midpoints: list[tuple[float, float]],
    distances: list[float],
) -> tuple[tuple[float, float], float] | None:
    """Centro (mediana de puntos medios) y escala (mediana de distancias) (§12.3).

    COMUNES a los 6 cuadros. Devuelve ``None`` si no hay ningún cuadro con
    hombros válidos (no se puede normalizar la secuencia).
    """
    if not midpoints or not distances:
        return None
    mx = float(np.median([m[0] for m in midpoints]))
    my = float(np.median([m[1] for m in midpoints]))
    scale = float(np.median(distances))
    if not math.isfinite(scale) or scale <= 0.0:
        return None
    return (mx, my), scale


def normalize_point(
    x_px: float,
    y_px: float,
    center: tuple[float, float],
    scale: float,
) -> tuple[float, float]:
    """``x_norm=(x_px-center_x)/scale``; igual en ``y`` (design §12.4)."""
    return (x_px - center[0]) / scale, (y_px - center[1]) / scale
