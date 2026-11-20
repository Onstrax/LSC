"""Aumentación de entrenamiento ``train_v2_robust`` (modelo v2).

Simula lo que la webcam hace distinto del corpus LSC70W (medido con 23 intentos reales, ver
``docs/models/v2_robusto.md``): la persona está más cerca, la mano que no firma queda **fuera del
cuadro** y una sola mano aparece en la mayoría de los cuadros. La política ``train_v1`` de Ortiz
(rotación ±5° y ruido 0,005) no cubre nada de eso, y se deja intacta.

Transformaciones (todas deterministas por semilla):
  1. afín: escala, traslación y rotación leves;
  2. ruido gaussiano y repetición de un cuadro (jitter temporal);
  3. recorte inferior y lateral (mano/brazo fuera de cuadro);
  4. una mano ausente en toda la secuencia, y pérdidas por cuadro.

Invariante: lo que sale **sigue siendo procesable** (cumple los criterios de calidad de la app). Si una
combinación lo rompe, se reintenta con menos intensidad; en el peor caso se devuelve solo ruido.
Nunca modifica la entrada. Solo se usa en entrenamiento; validación y producción no aumentan.
"""

from __future__ import annotations

import numpy as np

from . import perturb as P

POLICY_TRAIN_V2 = "train_v2_robust"

DEFAULT_PARAMS = {
    "p_affine": 0.5, "scale_range": (0.85, 1.2), "shift": 0.1, "rot_deg": 8.0,
    "p_noise": 0.5, "noise_std": 0.01,
    "p_temporal": 0.15,
    "p_crop_y": 0.5, "crop_y_range": (0.5, 1.3),
    "p_crop_x": 0.3, "crop_x_range": (0.9, 1.8),
    "p_drop_hand": 0.30,
    "p_frame_drop": 0.10,
    "max_retries": 4,
}


def _once(x: np.ndarray, rng: np.random.Generator, prm: dict, strength: float) -> np.ndarray:
    z = P.to_z(x)
    s = strength

    if rng.random() < prm["p_affine"] * s:
        lo, hi = prm["scale_range"]
        scale = 1.0 + (rng.uniform(lo, hi) - 1.0) * s
        P.affine(z, scale, rng.uniform(-prm["shift"], prm["shift"]) * s, rng.uniform(-prm["shift"], prm["shift"]) * s,
                 rng.uniform(-prm["rot_deg"], prm["rot_deg"]) * s)

    if rng.random() < prm["p_noise"]:
        present = z[..., 2] > 0
        z[..., 0:2] += (rng.normal(0.0, prm["noise_std"], z[..., 0:2].shape) * present[..., None]).astype(np.float32)

    if rng.random() < prm["p_temporal"] * s:
        f = int(rng.integers(0, P.T))
        g = min(P.T - 1, max(0, f + int(rng.choice((-1, 1)))))
        z[f] = z[g]

    if rng.random() < prm["p_crop_y"] * s:
        P.crop_below(z, float(rng.uniform(*prm["crop_y_range"])))
    if rng.random() < prm["p_crop_x"] * s:
        P.crop_sides(z, float(rng.uniform(*prm["crop_x_range"])))

    if rng.random() < prm["p_drop_hand"] * s:
        P.drop_hand(z, "left" if rng.random() < 0.5 else "right")
    for which in ("left", "right"):
        lost = np.flatnonzero(rng.random(P.T) < prm["p_frame_drop"] * s)
        if len(lost):
            P.drop_hand(z, which, lost)

    # ausente exacto en cero (también tras el ruido)
    z[z[..., 2] <= 0] = 0.0
    return P.to_x(z)


def augment_robust(x: np.ndarray, seed: int, params: dict | None = None) -> np.ndarray:
    """Devuelve una copia ``(6, 147)`` float32 aumentada y todavía procesable."""
    prm = {**DEFAULT_PARAMS, **(params or {})}
    rng = np.random.default_rng(seed)
    base = np.array(x, dtype=np.float32, copy=True).reshape(P.T, P.N_POINTS * 3)
    for k in range(int(prm["max_retries"]) + 1):
        strength = max(0.0, 1.0 - 0.3 * k)
        out = _once(base, rng, prm, strength)
        if P.is_processable(out):
            return out
    return _once(base, rng, {**prm, "p_affine": 0, "p_temporal": 0, "p_crop_y": 0, "p_crop_x": 0, "p_drop_hand": 0, "p_frame_drop": 0}, 0.0)
