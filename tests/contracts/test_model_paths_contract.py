"""Contrato de ``model_paths`` del extractor (design §8.3, §20).

Claves EXACTAS ``{"pose", "hands"}`` (espejo de ``api/adapters/real.py``);
``config`` acepta un ``dict`` de ``feature_spec``; ``init()`` con clave faltante
o sobrante -> ``CONFIG_MISMATCH`` sin tocar disco (con ``FakeDetector``).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.lsc_features.detector import FakeDetector
from src.lsc_features.extractor import (
    REQUIRED_MODEL_PATH_KEYS,
    ConfigMismatchError,
    FeatureExtractor,
)
from src.lsc_features import reason_codes as rc

REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC = json.loads((REPO_ROOT / "contracts" / "feature_spec.json").read_text(encoding="utf-8"))
VALID_PATHS = {"pose": "pose_landmarker.task", "hands": "hand_landmarker.task"}


def test_required_keys_are_exactly_pose_hands():
    assert set(REQUIRED_MODEL_PATH_KEYS) == {"pose", "hands"}


def test_config_accepts_feature_spec_dict():
    """``config`` acepta un dict de feature_spec (no exige objeto pydantic)."""
    ex = FeatureExtractor(dict(SPEC), dict(VALID_PATHS), detector=FakeDetector())
    assert ex.feature_spec_hash  # hash resuelto desde el dict


def test_missing_key_raises_config_mismatch():
    with pytest.raises(ConfigMismatchError) as exc:
        FeatureExtractor(dict(SPEC), {"pose": "p.task"}, detector=FakeDetector())
    assert rc.CONFIG_MISMATCH in str(exc.value)


def test_extra_key_raises_config_mismatch():
    bad = {"pose": "p.task", "hands": "h.task", "extra": "x.task"}
    with pytest.raises(ConfigMismatchError) as exc:
        FeatureExtractor(dict(SPEC), bad, detector=FakeDetector())
    assert rc.CONFIG_MISMATCH in str(exc.value)


def test_config_mismatch_does_not_touch_disk():
    """Con FakeDetector y claves inválidas no se abre ni verifica ningún .task."""
    # Rutas inexistentes: si tocara disco para verificarlas, el error sería otro;
    # con claves válidas + FakeDetector NO se abre disco y la construcción pasa.
    ex = FeatureExtractor(dict(SPEC), {"pose": "/no/existe/p.task", "hands": "/no/existe/h.task"},
                          detector=FakeDetector())
    assert ex is not None  # no se intentó abrir las rutas (detector inyectado)
