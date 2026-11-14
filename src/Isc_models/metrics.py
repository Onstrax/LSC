"""Métricas. Separa tres cosas que el artículo NO debe mezclar (plan §11):

1. desempeño sobre muestras PROCESABLES (``quality_pass=True``),
2. desempeño INTEGRAL: los rechazos de calidad cuentan como no acertados,
3. cobertura de calidad (fracción de muestras que el extractor deja pasar).

El rechazo de desconocidos (umbral/negativos) es A4 y se mide aparte.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

NUM_CLASSES = 10
LABELS = list(range(NUM_CLASSES))
RETRY = -1  # "predicción" de una muestra rechazada por calidad


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) == 0:
        return float("nan")
    _, _, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, average=None, zero_division=0
    )
    return float(np.mean(f1))


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) == 0:
        return float("nan")
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def cross_entropy(y_true: np.ndarray, probs: np.ndarray, eps: float = 1e-12) -> float:
    if len(y_true) == 0:
        return float("nan")
    p = np.clip(probs[np.arange(len(y_true)), y_true], eps, 1.0)
    return float(-np.mean(np.log(p)))


def per_class_table(y_true: np.ndarray, y_pred: np.ndarray) -> list[dict]:
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, average=None, zero_division=0
    )
    return [
        {"class_id": c, "precision": float(p[c]), "recall": float(r[c]), "f1": float(f[c]), "support": int(s[c])}
        for c in LABELS
    ]


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=LABELS)


def evaluate_split(y: np.ndarray, probs: np.ndarray, quality_pass: np.ndarray) -> dict:
    """Evalúa un split completo (procesables + integral)."""
    y = np.asarray(y)
    qp = np.asarray(quality_pass, dtype=bool)
    pred_all = np.argmax(probs, axis=1)
    pred_proc = pred_all[qp]
    y_proc = y[qp]
    pred_integral = np.where(qp, pred_all, RETRY)
    n = int(len(y))
    return {
        "n": n,
        "n_processable": int(qp.sum()),
        "quality_coverage": float(qp.mean()) if n else float("nan"),
        "accuracy_processable": accuracy(y_proc, pred_proc),
        "macro_f1_processable": macro_f1(y_proc, pred_proc),
        "loss_processable": cross_entropy(y_proc, probs[qp]),
        # integral: un rechazo de calidad NO es acierto (y no es una clase predicha)
        "accuracy_integral": accuracy(y, pred_integral),
        "macro_f1_integral": macro_f1(y, pred_integral),
    }


def p95(values) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), 95))
