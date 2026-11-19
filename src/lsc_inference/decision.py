"""Política de decisión sobre probabilidades (propiedad de Acosta).

Modo ``max_softmax_threshold``: se acepta la clase de mayor probabilidad si su
probabilidad es >= ``threshold`` (comparación exacta; desempate: menor índice).
Si no, se rechaza con ``LOW_CONFIDENCE``.

IMPORTANTE: la probabilidad softmax NO es certeza de que la seña sea correcta
(plan §5). ``requires_confirmation`` indica a la interfaz que debe presentar la
salida como sugerencia que el usuario confirma; el umbral final lo fija A4 con
validación y webcam de desarrollo, no se inventa aquí.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

SUPPORTED_MODES = ("max_softmax_threshold",)


@dataclass(frozen=True)
class Decision:
    accepted: bool
    label_id: int | None
    confidence: float | None
    reason_codes: list[str] = field(default_factory=list)


class DecisionPolicy:
    def __init__(self, policy: dict):
        mode = policy.get("mode", "max_softmax_threshold")
        if mode not in SUPPORTED_MODES:
            raise ValueError(f"modo de política no soportado: {mode}")
        if not policy.get("policy_version"):
            raise ValueError("la política requiere policy_version")
        thr = float(policy.get("threshold", 0.0))
        if not 0.0 <= thr <= 1.0:
            raise ValueError("threshold debe estar en [0, 1]")
        self.mode = mode
        self.threshold = thr
        self.policy_version = str(policy["policy_version"])
        self.requires_confirmation = bool(policy.get("requires_confirmation", True))

    def decide(self, probabilities) -> Decision:
        p = np.asarray(probabilities, dtype=np.float32)
        if p.ndim == 2:
            if p.shape[0] != 1:
                raise ValueError("decide() recibe una sola muestra")
            p = p[0]
        if p.shape != (10,) or not np.all(np.isfinite(p)):
            raise ValueError("probabilidades inválidas")
        label_id = int(np.argmax(p))
        conf = float(p[label_id])
        if conf >= self.threshold:
            return Decision(True, label_id, conf, [])
        return Decision(False, None, None, ["LOW_CONFIDENCE"])
