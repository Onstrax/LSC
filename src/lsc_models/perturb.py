"""Perturbaciones deterministas sobre secuencias de características ``(6, 147)``.

Las comparten la aumentación de entrenamiento v2 (``augment_robust``) y la batería de robustez
sobre validación (``robustness``): así lo que se entrena y lo que se mide hablan del mismo fallo.

Convenciones (``contracts/feature_spec.json``): 6 instantes x 49 puntos x ``(x, y, m)``; un punto
ausente es ``x = y = m = 0``; ``y`` crece hacia abajo; coordenadas normalizadas por el punto medio y la
distancia de hombros. Orden de puntos: ``pose_0, pose_11, pose_12, pose_13, pose_14, pose_15, pose_16``
(0..6), mano izquierda ``lh_0..20`` (7..27) y mano derecha ``rh_0..20`` (28..48).

Una mano es un bloque: en una captura real la mano se detecta entera o no se detecta; por eso los
recortes de cuadro actúan sobre la muñeca de la mano (punto 0 de su bloque) y la quitan completa.
"""

from __future__ import annotations

import math

import numpy as np

T, N_POINTS = 6, 49
POSE_SHOULDERS = (1, 2)
POSE_ARM = (3, 4, 5, 6)  # codos y muñecas de la pose: se recortan por su propia coordenada
HAND_SLICES = {"left": slice(7, 28), "right": slice(28, 49)}
HAND_WRIST = {"left": 7, "right": 28}
# Criterios de calidad de la app (contracts/feature_spec.json, quality_rules)
MIN_SHOULDER_FRAMES = 5
MIN_HAND_FRAMES = 4


def to_z(x: np.ndarray) -> np.ndarray:
    """``(6, 147)`` -> copia ``(6, 49, 3)`` float32."""
    return np.array(x, dtype=np.float32, copy=True).reshape(T, N_POINTS, 3)


def to_x(z: np.ndarray) -> np.ndarray:
    return z.reshape(T, N_POINTS * 3).astype(np.float32, copy=False)


def _clear(z: np.ndarray, frame: int | slice, points: slice | tuple) -> None:
    z[frame, points, :] = 0.0


def drop_hand(z: np.ndarray, which: str, frames=None) -> np.ndarray:
    """Quita la mano ``which`` ('left'|'right') en ``frames`` (None = todos). Opera in situ."""
    sl = HAND_SLICES[which]
    if frames is None:
        _clear(z, slice(None), sl)
    else:
        for f in frames:
            _clear(z, int(f), sl)
    return z


def crop_below(z: np.ndarray, t_y: float) -> np.ndarray:
    """Fuera de cuadro por abajo: manos con muñeca ``y > t_y`` y codos/muñecas de pose con ``y > t_y``."""
    for which in HAND_SLICES:
        w = HAND_WRIST[which]
        out = (z[:, w, 2] > 0) & (z[:, w, 1] > t_y)
        for f in np.flatnonzero(out):
            _clear(z, int(f), HAND_SLICES[which])
    for p in POSE_ARM:
        out = (z[:, p, 2] > 0) & (z[:, p, 1] > t_y)
        z[out, p, :] = 0.0
    return z


def crop_sides(z: np.ndarray, t_x: float) -> np.ndarray:
    """Fuera de cuadro por los lados: ``|x| > t_x`` (misma regla que ``crop_below``)."""
    for which in HAND_SLICES:
        w = HAND_WRIST[which]
        out = (z[:, w, 2] > 0) & (np.abs(z[:, w, 0]) > t_x)
        for f in np.flatnonzero(out):
            _clear(z, int(f), HAND_SLICES[which])
    for p in POSE_ARM:
        out = (z[:, p, 2] > 0) & (np.abs(z[:, p, 0]) > t_x)
        z[out, p, :] = 0.0
    return z


def affine(z: np.ndarray, scale: float = 1.0, shift_x: float = 0.0, shift_y: float = 0.0, deg: float = 0.0) -> np.ndarray:
    """Escala + rotación (alrededor del origen) + traslación sobre los puntos presentes."""
    a = math.radians(deg)
    c, s = math.cos(a) * scale, math.sin(a) * scale
    present = z[..., 2] > 0
    x, y = z[..., 0].copy(), z[..., 1].copy()
    nx = x * c - y * s + shift_x
    ny = x * s + y * c + shift_y
    z[..., 0] = np.where(present, nx, 0.0)
    z[..., 1] = np.where(present, ny, 0.0)
    return z


def hand_frames(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cuadros (bool, longitud 6) con mano izquierda / derecha presente."""
    z = np.asarray(x, dtype=np.float32).reshape(T, N_POINTS, 3)
    return (z[:, HAND_SLICES["left"], 2] > 0).any(1), (z[:, HAND_SLICES["right"], 2] > 0).any(1)


def is_processable(x: np.ndarray) -> bool:
    """¿Pasaría la calidad de la app? (>= 5 cuadros con ambos hombros y >= 1 mano en >= 4 cuadros)."""
    z = np.asarray(x, dtype=np.float32).reshape(T, N_POINTS, 3)
    shoulders = ((z[:, POSE_SHOULDERS[0], 2] > 0) & (z[:, POSE_SHOULDERS[1], 2] > 0)).sum()
    left, right = hand_frames(z)
    return bool(shoulders >= MIN_SHOULDER_FRAMES and (left | right).sum() >= MIN_HAND_FRAMES)
