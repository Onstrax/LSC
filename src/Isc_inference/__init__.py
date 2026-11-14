"""Inferencia del proyecto LSC.

API pública consumida por ``api/adapters/real.py``:

    from src.lsc_inference import Predictor, DecisionPolicy

``Predictor(model_path).predict(x)``: float32 (B,6,147) -> float32 (B,10).
``DecisionPolicy(policy_dict).decide(probs)``: probabilidades -> decisión.

Importar el paquete NO carga TensorFlow (carga perezosa en ``Predictor``).
"""

from __future__ import annotations

from .decision import Decision, DecisionPolicy
from .predictor import Predictor, PredictorInputError, SimulatedPredictor

__all__ = ["Predictor", "SimulatedPredictor", "PredictorInputError", "DecisionPolicy", "Decision"]
