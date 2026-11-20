"""Regresión: RealExtractor debe inyectar un detector real al FeatureExtractor.

Con detector=None, FeatureExtractor solo verifica que existan los .task y falla al extraer
(AttributeError -> 500 INTERNAL_ERROR en /predict). Aquí se usan dobles: sin MediaPipe ni pesos.
"""

from __future__ import annotations

import json

import pytest

from api.adapters import real


def _artifacts(tmp_path):
    (tmp_path / "feature_spec.json").write_text(json.dumps({"feature_spec_hash": "h" * 64}), encoding="utf-8")
    (tmp_path / "pose_landmarker.task").write_bytes(b"pose")
    (tmp_path / "hand_landmarker.task").write_bytes(b"hands")
    return tmp_path


def test_real_extractor_injects_the_detector(tmp_path, monkeypatch):
    seen = {}

    class FakeDetector:
        def __init__(self, pose_path, hands_path):
            seen["paths"] = (str(pose_path), str(hands_path))

    class FakeFeatureExtractor:
        def __init__(self, config, model_paths, detector=None):
            seen["detector"] = detector

    import src.lsc_features as lf
    import src.lsc_features.detector as det

    monkeypatch.setattr(lf, "FeatureExtractor", FakeFeatureExtractor)
    monkeypatch.setattr(det, "RealDetector", FakeDetector)

    art = _artifacts(tmp_path)
    real.RealExtractor(art)

    assert isinstance(seen["detector"], FakeDetector)  # nunca None
    assert seen["paths"][0].endswith("pose_landmarker.task")
    assert seen["paths"][1].endswith("hand_landmarker.task")


def test_detector_creation_failure_is_an_explicit_artifact_error(tmp_path, monkeypatch):
    class BrokenDetector:
        def __init__(self, *_):
            raise RuntimeError("mediapipe no disponible")

    import src.lsc_features as lf
    import src.lsc_features.detector as det

    monkeypatch.setattr(lf, "FeatureExtractor", lambda *a, **k: None)
    monkeypatch.setattr(det, "RealDetector", BrokenDetector)

    with pytest.raises(real.ArtifactError, match="detector real de MediaPipe"):
        real.RealExtractor(_artifacts(tmp_path))
