"""Batería de robustez sobre VALIDACIÓN (nunca test): ¿qué pasa cuando falta una mano o hay recorte?

Cada escenario perturba las muestras procesables de validación de forma **determinista** (misma
semilla => mismo resultado) y se evalúa solo sobre las que la app aún aceptaría tras la perturbación
(``perturb.is_processable``). Se usa para tres cosas:

1. elegir la época y la configuración de v2 (``val_mixed_f1`` = media de F1 original y F1 robusto);
2. comparar v1 con v2 con la misma vara (comando ``robustness``);
3. documentar qué degrada al modelo, sin abrir el test.

Los escenarios reproducen lo medido con webcam (mano en reposo fuera del cuadro, una sola mano), pero
las capturas de la persona NO entran aquí: solo el corpus de validación.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable

import numpy as np

from . import metrics as M
from . import perturb as P

SCENARIO_NAMES = (
    "sin_mano_izq", "sin_mano_der", "recorte_1.0", "recorte_0.7", "recorte_0.5",
    "manos_por_tramos", "escala_traslacion", "webcam_like",
)
REFERENCE_THRESHOLD = 0.94  # umbral de v1; solo como vara común de comparación


def _apply(name: str, z: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if name == "sin_mano_izq":
        return P.drop_hand(z, "left")
    if name == "sin_mano_der":
        return P.drop_hand(z, "right")
    if name.startswith("recorte_"):
        return P.crop_below(z, float(name.split("_")[1]))
    if name == "manos_por_tramos":
        which = "left" if rng.random() < 0.5 else "right"
        return P.drop_hand(z, which, rng.choice(P.T, size=3, replace=False))
    if name == "escala_traslacion":
        return P.affine(z, scale=1.2, shift_x=0.1, shift_y=-0.1, deg=5.0)
    if name == "webcam_like":  # una mano ausente toda la secuencia + lo bajo fuera de cuadro
        P.drop_hand(z, "left" if rng.random() < 0.5 else "right")
        return P.crop_below(z, 0.7)
    raise ValueError(f"escenario desconocido: {name}")


def build_scenarios(X: np.ndarray, seed: int = 20261113) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """``{escenario: (X_perturbado, aceptable)}``; ``aceptable[i]`` = la app aún aceptaría la muestra i."""
    out = {}
    for k, name in enumerate(SCENARIO_NAMES):
        rng = np.random.default_rng(np.random.SeedSequence([seed, k]))
        Xp = np.stack([P.to_x(_apply(name, P.to_z(x), rng)) for x in X]).astype(np.float32)
        ok = np.array([P.is_processable(x) for x in Xp], dtype=bool)
        out[name] = (Xp, ok)
    return out


def evaluate_scenarios(predict: Callable[[np.ndarray], np.ndarray], y: np.ndarray,
                       scenarios: dict[str, tuple[np.ndarray, np.ndarray]],
                       threshold: float = REFERENCE_THRESHOLD) -> dict:
    """Métricas por escenario y medias. ``predict`` mapea ``(n, 6, 147)`` a probabilidades ``(n, 10)``."""
    per = {}
    # una sola llamada al modelo para todos los escenarios (más rápido por época)
    chunks = [Xp[ok] for Xp, ok in scenarios.values() if ok.any()]
    all_p = np.asarray(predict(np.concatenate(chunks)), dtype=np.float32) if chunks else np.zeros((0, M.NUM_CLASSES), np.float32)
    cursor = 0
    for name, (Xp, ok) in scenarios.items():
        n = int(ok.sum())
        if n == 0:
            per[name] = {"n": 0, "accuracy": float("nan"), "macro_f1": float("nan"), "conf_median": float("nan"), "frac_above_threshold": float("nan")}
            continue
        p = all_p[cursor:cursor + n]
        cursor += n
        pred, conf = p.argmax(1), p.max(1)
        per[name] = {
            "n": n, "accuracy": M.accuracy(y[ok], pred), "macro_f1": M.macro_f1(y[ok], pred),
            "conf_median": float(np.median(conf)), "frac_above_threshold": float((conf >= threshold).mean()),
        }
    valid = [v for v in per.values() if v["n"] > 0]
    keys = ("accuracy", "macro_f1", "conf_median", "frac_above_threshold")
    mean = {k: float(np.mean([v[k] for v in valid])) if valid else float("nan") for k in keys}
    return {"scenarios": per, "mean": mean, "threshold": threshold}


def mixed_f1(f1_original: float, robust: dict) -> float:
    """Criterio de selección de v2: media de F1 macro en validación original y en validación perturbada.

    Si ningún escenario tiene muestras aceptables (no debería pasar con datos reales) se usa solo el F1 original,
    para que el entrenamiento nunca compare contra NaN.
    """
    r = robust["mean"]["macro_f1"]
    return float(f1_original) if not np.isfinite(r) else 0.5 * float(f1_original) + 0.5 * float(r)


def compare(runs_by_label: dict[str, list[Path]], X: np.ndarray, y: np.ndarray, out_csv: Path | None = None,
            seed: int = 20261113, threshold: float = REFERENCE_THRESHOLD) -> list[dict]:
    """Evalúa modelos guardados (``model.keras``) en original + escenarios. ``runs_by_label``: etiqueta -> carpetas."""
    import keras

    from . import custom_layers  # noqa: F401  (registra PositionalEmbedding)

    scen = build_scenarios(X, seed)
    base_ok = np.ones(len(X), dtype=bool)
    rows = []
    for label, run_dirs in runs_by_label.items():
        for d in run_dirs:
            model = keras.models.load_model(Path(d) / "model.keras", compile=False)

            def predict(a, _m=model):
                return _m.predict(a, batch_size=64, verbose=0)

            res = evaluate_scenarios(predict, y, {"original": (X, base_ok), **scen}, threshold)
            for sc, v in res["scenarios"].items():
                rows.append({"grupo": label, "ejecucion": Path(d).name, "escenario": sc, **v})
            rows.append({"grupo": label, "ejecucion": Path(d).name, "escenario": "MEDIA_ROBUSTA",
                         "n": "", **{k: res["mean"][k] for k in ("accuracy", "macro_f1", "conf_median", "frac_above_threshold")}})
    if out_csv is not None and rows:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    return rows


def summarize_rows(rows: list[dict]) -> list[dict]:
    """Media por (grupo, escenario) sobre ejecuciones, lista para imprimir."""
    out = {}
    for r in rows:
        out.setdefault((r["grupo"], r["escenario"]), []).append(r)
    res = []
    for (g, sc), rs in out.items():
        res.append({"grupo": g, "escenario": sc, "n_runs": len(rs),
                    **{k: float(np.mean([r[k] for r in rs])) for k in ("accuracy", "macro_f1", "conf_median", "frac_above_threshold")}})
    return res
