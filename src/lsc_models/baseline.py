"""Clasificador trivial de clase mayoritaria (contexto de las métricas, plan §6).

Se ajusta SOLO con las muestras procesables de train. Ante empate en la clase más
frecuente gana el menor ``class_id`` (determinista). Produce "probabilidades"
one-hot para poder usar la misma función de evaluación que los modelos.
"""

from __future__ import annotations

import numpy as np

from .metrics import NUM_CLASSES


class MajorityClassifier:
    def __init__(self) -> None:
        self.majority_class: int | None = None
        self.class_counts: list[int] = []

    def fit(self, y_train: np.ndarray) -> "MajorityClassifier":
        counts = np.bincount(np.asarray(y_train, dtype=np.int64), minlength=NUM_CLASSES)
        self.class_counts = counts.tolist()
        self.majority_class = int(np.argmax(counts))  # menor índice en empate
        return self

    def predict_proba(self, n: int) -> np.ndarray:
        if self.majority_class is None:
            raise RuntimeError("fit() no se ha llamado")
        probs = np.zeros((n, NUM_CLASSES), dtype=np.float32)
        probs[:, self.majority_class] = 1.0
        return probs
