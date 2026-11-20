"""Sensibilidad a fallos del extractor (A3) — SOLO validación, SOLO diagnóstico.

Dos análisis, ambos sobre los modelos ya entrenados (no se reentrena ni se ajusta
nada con ellos):

1. Contabilidad de calidad: cuántas muestras de validación rechaza el extractor
   por clase y cuánto cae el desempeño integral respecto al de procesables.
2. Barrido de pérdida de manos: en las muestras procesables se marcan como
   ausentes (ceros, máscara 0) los puntos de las manos en una fracción de los
   instantes, y se mide la caída de exactitud/F1. Las perturbaciones se aplican
   solo en evaluación y no alteran etiquetas, particiones ni pesos.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from . import metrics as M
from .config import REQUIRED_CONFIGS, REQUIRED_SEEDS
from .data import Bundle

# point_order del feature_spec: pose_* = 0..6, lh_* = 7..27, rh_* = 28..48
HAND_SLICES = {"both": range(7, 49), "left": range(7, 28), "right": range(28, 49)}


def drop_hands(X: np.ndarray, fraction: float, which: str, rng: np.random.Generator) -> np.ndarray:
    out = X.copy()
    n, T, _ = out.shape
    k = int(round(fraction * T))
    pts = list(HAND_SLICES[which])
    for i in range(n):
        if k == 0:
            continue
        frames = rng.choice(T, size=k, replace=False)
        for t in frames:
            for pt in pts:
                out[i, t, pt * 3 : pt * 3 + 3] = 0.0
    return out


def quality_accounting(bundle: Bundle) -> list[dict]:
    va = bundle.load_split("val")
    rows = []
    for c in range(M.NUM_CLASSES):
        m = va.y == c
        rows.append(
            {
                "class_id": c,
                "n_val": int(m.sum()),
                "n_rejected_quality": int((m & ~va.quality_pass).sum()),
                "rejected_fraction": float((m & ~va.quality_pass).sum() / m.sum()) if m.sum() else float("nan"),
            }
        )
    return rows


def run_sensitivity(bundle: Bundle, runs_dir: str | Path, out_dir: str | Path, cfg: dict, *, fast: bool = False,
                    names: list[str] | None = None) -> Path:
    import keras

    from . import custom_layers  # noqa: F401  (registra PositionalEmbedding)
    from .config import FAST_OVERRIDES, ConfigError

    if fast and not bundle.synthetic:
        raise ConfigError("--fast solo se permite con datos sintéticos")
    scfg = FAST_OVERRIDES["sensitivity"] | {"seed": cfg["sensitivity"]["seed"]} if fast else cfg["sensitivity"]
    va = bundle.load_split("val")
    p = va.processable()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in (names or REQUIRED_CONFIGS):
        for seed in REQUIRED_SEEDS:
            model = keras.models.load_model(Path(runs_dir) / f"{name}__seed{seed}" / "model.keras", compile=False)
            for which in ("both", "left", "right"):
                for frac in scfg["hand_dropout_fractions"]:
                    accs, f1s = [], []
                    reps = 1 if frac == 0.0 else int(scfg["repetitions"])
                    for rep in range(reps):
                        rng = np.random.default_rng(np.random.SeedSequence([int(scfg["seed"]), seed, rep, int(frac * 100)]))
                        Xp = drop_hands(p.X, frac, which, rng)
                        pred = np.argmax(model.predict(Xp, batch_size=64, verbose=0), 1)
                        accs.append(M.accuracy(p.y, pred))
                        f1s.append(M.macro_f1(p.y, pred))
                    rows.append(
                        {
                            "config": name, "seed": seed, "hands_dropped": which, "frame_fraction": frac,
                            "accuracy_mean": float(np.mean(accs)), "macro_f1_mean": float(np.mean(f1s)),
                            "repetitions": reps, "n_eval": int(len(p)),
                        }
                    )
    path = out / "sensitivity_hand_dropout.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    qa = quality_accounting(bundle)
    with (out / "quality_rejections_val.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(qa[0].keys()))
        w.writeheader()
        w.writerows(qa)
    (out / "sensitivity_meta.json").write_text(
        json.dumps({"scope": "validación únicamente", "diagnostic_only": True, "evidence_valid": not bundle.synthetic}, indent=2) + "\n",
        encoding="utf-8",
    )
    return path
