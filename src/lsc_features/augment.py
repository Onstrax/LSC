"""Augmentation de entrenamiento (design §14).

``augment_train(x, seed, policy)`` opera sobre una copia float32, preserva forma
y máscaras, sólo modifica ``x,y`` donde ``m=1`` y lo llama únicamente el loader
de entrenamiento. Política ``train_v1``: rotación global por secuencia (prob 0.5,
ángulo uniforme [-5°, +5°] alrededor del origen) y luego ruido gaussiano
(std 0.005, prob 0.5). Sin reflexión ni inversión temporal. Val/test/producción
NUNCA aplican augmentation.

El array ``x`` tiene forma ``(6, 147)`` = 6 instantes x 49 triples ``(x, y, m)``.
"""

from __future__ import annotations

import math

import numpy as np

POLICY_TRAIN_V1 = "train_v1"
_ROT_PROB = 0.5
_ROT_DEG = 5.0
_NOISE_PROB = 0.5
_NOISE_STD = 0.005


def augment_train(x: np.ndarray, seed: int, policy: str = POLICY_TRAIN_V1) -> np.ndarray:
    """Devuelve una copia float32 de ``x`` con la política aplicada (design §14).

    Sólo la política ``train_v1`` modifica los datos. ``val``/``test``/
    ``production`` (o cualquier política distinta de ``train_v1``) devuelven una
    copia sin cambios. Nunca modifica el array de entrada ni el archivo base.
    """
    out = np.array(x, dtype=np.float32, copy=True)
    if policy != POLICY_TRAIN_V1:
        return out

    rng = np.random.default_rng(seed)

    # Reshape a (frames, 49, 3) -> canales (x, y, m).
    frames = out.reshape(out.shape[0], -1, 3)
    xs = frames[..., 0]
    ys = frames[..., 1]
    mask = frames[..., 2] == 1.0

    # 1) Rotación global por secuencia (mismo ángulo para todos los cuadros).
    if rng.random() < _ROT_PROB:
        angle = math.radians(float(rng.uniform(-_ROT_DEG, _ROT_DEG)))
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        new_x = xs * cos_a - ys * sin_a
        new_y = xs * sin_a + ys * cos_a
        xs[mask] = new_x[mask]
        ys[mask] = new_y[mask]

    # 2) Ruido gaussiano independiente (sólo donde m=1).
    if rng.random() < _NOISE_PROB:
        noise_x = rng.normal(0.0, _NOISE_STD, size=xs.shape).astype(np.float32)
        noise_y = rng.normal(0.0, _NOISE_STD, size=ys.shape).astype(np.float32)
        xs[mask] = xs[mask] + noise_x[mask]
        ys[mask] = ys[mask] + noise_y[mask]

    # Garantizar que los ceros de ausencia siguen en cero.
    xs[~mask] = 0.0
    ys[~mask] = 0.0

    return out.astype(np.float32, copy=False)
