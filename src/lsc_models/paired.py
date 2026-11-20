"""Comparación PAREADA de dos modelos ya evaluados en test (v1 frente a v2).

Lee SOLO las predicciones de test que `final-eval` ya guardó (``predictions_test_deployed.npz``) y no vuelve a
evaluar ningún modelo ni toca el split. Reporta aciertos de cada uno, discordantes (solo uno acierta), el test
exacto de McNemar (bilateral) y un IC95 % bootstrap de la diferencia de exactitud sobre las muestras procesables.
Con n pequeño (131) conviene mirar esto y no solo la diferencia de cifras.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


def mcnemar_exact(only_a: int, only_b: int) -> float:
    """p-valor exacto bilateral para ``only_a`` vs ``only_b`` discordantes."""
    k = only_a + only_b
    if k == 0:
        return 1.0
    m = min(only_a, only_b)
    return float(min(1.0, 2.0 * sum(math.comb(k, i) for i in range(m + 1)) / 2 ** k))


def paired_comparison(a_npz: str | Path, b_npz: str | Path, *, n_boot: int = 5000, seed: int = 0) -> dict:
    a = np.load(a_npz, allow_pickle=False)
    b = np.load(b_npz, allow_pickle=False)
    if not (np.array_equal(a["sample_id"], b["sample_id"]) and np.array_equal(a["y_true"], b["y_true"])):
        raise ValueError("las predicciones no son sobre las mismas muestras: no se pueden parear")
    q = np.asarray(a["quality_pass"], dtype=bool)
    y = a["y_true"][q]
    ca, cb = a["probs"][q].argmax(1) == y, b["probs"][q].argmax(1) == y
    n = int(len(y))
    only_b, only_a = int((cb & ~ca).sum()), int((ca & ~cb).sum())
    rng = np.random.default_rng(seed)
    diffs = np.array([(lambda i: cb[i].mean() - ca[i].mean())(rng.integers(0, n, n)) for _ in range(n_boot)])
    return {
        "n_processable": n,
        "correct_a": int(ca.sum()), "correct_b": int(cb.sum()),
        "both_correct": int((ca & cb).sum()), "only_a_correct": only_a, "only_b_correct": only_b,
        "none_correct": int((~ca & ~cb).sum()),
        "accuracy_a": float(ca.mean()), "accuracy_b": float(cb.mean()),
        "difference_b_minus_a": float(cb.mean() - ca.mean()),
        "bootstrap_ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
        "mcnemar_exact_p_two_sided": mcnemar_exact(only_a, only_b),
        "n_bootstrap": n_boot, "bootstrap_seed": seed,
    }


def write_paired_comparison(v1_reports: str | Path, v2_reports: str | Path, out_json: str | Path | None = None) -> Path:
    v1, v2 = Path(v1_reports), Path(v2_reports)
    res = paired_comparison(v1 / "predictions_test_deployed.npz", v2 / "predictions_test_deployed.npz")
    doc = {
        "schema_version": "lsc.paired_test_comparison.v1",
        "a": "v1 (reports/models, modelo desplegado de v1)",
        "b": "v2 (reports/models_v2, modelo desplegado de v2)",
        "source": "predictions_test_deployed.npz de cada carpeta (guardadas por final-eval; no se reevaluó nada)",
        "note": "Test usado dos veces: primero v1, luego v2 (ver reports/models_v2/test_authorization.json).",
        **res,
    }
    out = Path(out_json) if out_json else v2 / "paired_test_comparison.json"
    out.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out
