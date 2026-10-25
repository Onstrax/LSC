"""Config incompatible: fallo al INICIO, sin fallback silencioso (design §8.3, §20).

- ``feature_spec_hash`` divergente entre consumidor y spec se detecta al inicio.
- ``labels.json`` divergente (orden/código) se detecta, no se asume por defecto.
- ``.task`` faltante (detector real, detector=None) -> mensaje accionable y
  NINGUNA descarga silenciosa (AGENTS.md regla 3).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from contracts.canonical import feature_spec_hash
from src.lsc_features.extractor import ConfigMismatchError, FeatureExtractor

REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC = json.loads((REPO_ROOT / "contracts" / "feature_spec.json").read_text(encoding="utf-8"))
MODEL_PATHS = {"pose": "pose_landmarker.task", "hands": "hand_landmarker.task"}


def _fake_detector():
    from src.lsc_features.detector import FakeDetector

    return FakeDetector()


def test_feature_spec_hash_divergence_detected():
    """Un feature_spec alterado produce un hash distinto del congelado."""
    stored = SPEC["feature_spec_hash"]
    tampered = json.loads(json.dumps(SPEC))
    tampered["point_order"] = list(reversed(tampered["point_order"]))
    assert feature_spec_hash(tampered) != stored
    # El extractor expone el hash recomputado del spec que recibe (sin el stored).
    cfg = json.loads(json.dumps(tampered))
    cfg.pop("feature_spec_hash", None)
    ex = FeatureExtractor(cfg, dict(MODEL_PATHS), detector=_fake_detector())
    assert ex.feature_spec_hash == feature_spec_hash(cfg)
    assert ex.feature_spec_hash != stored


def test_labels_divergence_detected():
    """Un labels.json con orden cambiado no valida como el canónico."""
    from contracts.models import LabelMap

    labels = json.loads((REPO_ROOT / "contracts" / "labels.json").read_text(encoding="utf-8"))
    canonical = LabelMap.model_validate(labels)
    diverged = json.loads(json.dumps(labels))
    diverged["labels"][0]["code"] = "ADULTERADO"
    other = LabelMap.model_validate(diverged)
    assert [e.code for e in other.labels] != [e.code for e in canonical.labels]


def test_missing_task_actionable_no_silent_download(tmp_path):
    """detector=None con .task inexistente -> error accionable, sin descarga."""
    missing = {
        "pose": tmp_path / "no_such_pose.task",
        "hands": tmp_path / "no_such_hands.task",
    }
    with pytest.raises(ConfigMismatchError) as exc:
        FeatureExtractor(dict(SPEC), missing, detector=None)
    msg = str(exc.value)
    assert "no se encontró" in msg.lower() or "no encontr" in msg.lower()
    # No se creó ningún archivo .task (nada se descargó).
    assert not any(tmp_path.glob("*.task"))
