"""Exportación de artefactos + arranque de la API en modo REAL (solo el extractor MediaPipe se simula,
porque necesita los .task). Datos sintéticos: valida integración, NO reconocimiento de LSC."""

import io
import json
import uuid

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from src.lsc_models.config import FAST_OVERRIDES
from src.lsc_models.data import Bundle, make_synthetic_contract_fixture
from src.lsc_models.experiments import run_all
from src.lsc_models.export import ExportError, export_artifacts, fit_threshold

RUN = "bilstm_noaug__seed17"


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    root = tmp_path_factory.mktemp("exp")
    bundle = make_synthetic_contract_fixture(root / "syn" / "features.npz", n_signers=20)
    run_all("configs/train/experiments.yaml", root / "rep", bundle_path=bundle, only=[RUN], overrides=FAST_OVERRIDES)
    return {"root": root, "bundle": bundle, "rep": root / "rep"}


@pytest.fixture(scope="module")
def artifacts(env):
    out = env["root"] / "artifacts"
    export_artifacts(env["rep"], out, env["bundle"], run_id=RUN, allow_synthetic=True, models_dir=env["root"] / "no_models")
    return out


def test_export_refuses_synthetic_by_default(env, tmp_path):
    with pytest.raises(ExportError, match="sintéticos"):
        export_artifacts(env["rep"], tmp_path / "a", env["bundle"], run_id=RUN)


def test_export_refuses_overwrite_without_force(env, artifacts):
    with pytest.raises(ExportError, match="--force"):
        export_artifacts(env["rep"], artifacts, env["bundle"], run_id=RUN, allow_synthetic=True)


def test_export_refuses_wrong_bundle(env, tmp_path):
    other = make_synthetic_contract_fixture(tmp_path / "other.npz", n_signers=14)
    with pytest.raises(ExportError, match="SHA-256 distinto"):
        export_artifacts(env["rep"], tmp_path / "a", other, run_id=RUN, allow_synthetic=True)


def test_package_files_and_manifest(artifacts):
    for f in ("model.keras", "labels.json", "feature_spec.json", "preprocessing.json", "thresholds.json", "model_card.md", "manifest.json"):
        assert (artifacts / f).is_file(), f
    m = json.loads((artifacts / "manifest.json").read_text())
    spec = json.loads((artifacts / "feature_spec.json").read_text())
    assert m["feature_spec_hash"] == spec["feature_spec_hash"]
    assert m["synthetic_demo"] is True and m["evidence_valid"] is False
    assert m["model_version"].startswith("SYNTHETIC-DEMO-")
    assert m["parity_max_abs_diff"] <= 1e-5 and m["test_used"] is False
    import hashlib

    for f, h in m["files_sha256"].items():
        assert hashlib.sha256((artifacts / f).read_bytes()).hexdigest() == h
    assert set(m["missing_extractor_assets"]) == {"pose_landmarker.task", "hand_landmarker.task"}  # no se inventan


def test_labels_are_texts_for_the_api_and_keep_contract_entries(artifacts):
    d = json.loads((artifacts / "labels.json").read_text(encoding="utf-8"))
    assert d["labels"][:3] == ["Hola", "Buenos", "Días"] and len(d["labels"]) == 10
    assert d["entries"][0]["class_id"] == 0 and d["entries"][0]["code"] == "hola"


def test_threshold_policy_is_provisional_and_requires_confirmation(artifacts):
    t = json.loads((artifacts / "thresholds.json").read_text())
    assert t["provisional"] is True and t["requires_confirmation"] is True
    assert t["negatives_rejection_measured"] is False and t["fit"]["coverage"] >= 0.8


def test_fit_threshold_strictest_with_coverage():
    y = np.array([0, 1, 2, 3, 4])
    probs = np.eye(10, dtype=np.float32)[y] * 0.0 + 0.01
    for i, c in enumerate([0.9, 0.8, 0.7, 0.6, 0.5]):
        probs[i, y[i]] = c
    probs[4, 4], probs[4, 0] = 0.3, 0.5  # la más baja confianza se equivoca
    f = fit_threshold(y, probs, np.ones(5, bool))
    assert f["threshold"] == pytest.approx(0.6) and f["coverage"] == pytest.approx(0.8) and f["accuracy_accepted"] == 1.0


def test_fit_threshold_needs_processable_samples():
    with pytest.raises(ExportError):
        fit_threshold(np.array([0]), np.eye(10, dtype=np.float32)[:1], np.array([False]))


# --------------------------------------------------------------------------- modo REAL
def _jpeg(i):
    buf = io.BytesIO()
    Image.new("RGB", (320, 240), (90 + i, 120, 150)).save(buf, "JPEG")
    return buf.getvalue()


def _meta(proto):
    return {"schema_version": "lsc.http.v1", "client_request_id": str(uuid.uuid4()), "capture_protocol_version": proto,
            "source": "authorized_fixture", "width": 320, "height": 240, "frame_count": 6, "observed_frame_count": 6,
            "selected_indices": list(range(6)), "timestamps_ms": [0, 200, 400, 600, 800, 1000],
            "capture_duration_ms": 1000, "mirrored": False}


def _real_client(monkeypatch, artifacts, x_fixed, with_extractor=True):
    from api.adapters import real
    from api.adapters.types import FeatureResult

    monkeypatch.setenv("LSC_MODE", "real")
    monkeypatch.setenv("LSC_ARTIFACTS_DIR", str(artifacts))
    monkeypatch.setenv("LSC_ALLOWED_HOSTS", "localhost,127.0.0.1,testserver")
    if with_extractor:

        class StubExtractor:  # reemplaza SOLO a MediaPipe; el resto del camino es real
            def __init__(self, artifacts_dir, feature_config=None):
                self.feature_spec_hash = json.loads((artifacts_dir / "feature_spec.json").read_text())["feature_spec_hash"]

            def extract_sequence(self, frames):
                return FeatureResult(x=x_fixed, quality_pass=True, reason_codes=[], valid_shoulders=True,
                                     frames_with_hand=6, feature_spec_hash=self.feature_spec_hash, elapsed_ms=0.0)

        monkeypatch.setattr(real, "RealExtractor", StubExtractor)
    import api.main as main_module

    return TestClient(main_module.app)


def test_real_mode_full_request_matches_direct_prediction(env, artifacts, monkeypatch):
    from src.lsc_inference import Predictor

    x = Bundle(env["bundle"]).load_split("val").processable().X[0]
    direct = Predictor(artifacts / "model.keras").predict(x)[0]
    with _real_client(monkeypatch, artifacts, x) as c:
        h = c.get("/api/v1/health")
        assert h.status_code == 200 and h.json()["ready"] and h.json()["mode"] == "real"
        assert h.json()["model_version"].startswith("SYNTHETIC-DEMO-")
        cfg = c.get("/api/v1/config").json()
        assert all(isinstance(t, str) for t in cfg["labels"])
        files = [("frames", (f"frame_{i}.jpg", _jpeg(i), "image/jpeg")) for i in range(6)]
        r = c.post("/api/v1/predict", files=files, data={"metadata": json.dumps(_meta(cfg["capture_protocol"]["capture_protocol_version"]))})
    body = r.json()
    assert r.status_code == 200 and body["mode"] == "real" and body["status"] in ("accepted", "retry")
    if body["status"] == "accepted":
        assert body["label_id"] == int(direct.argmax()) and isinstance(body["display_text"], str)
        assert body["confidence"] == pytest.approx(float(direct.max()), abs=1e-5)
    else:
        assert "LOW_CONFIDENCE" in body["reason_codes"] and body["label_id"] is None


def test_real_mode_incompatible_hash_is_503_not_mock(env, artifacts, monkeypatch, tmp_path):
    import shutil

    bad = tmp_path / "bad"
    shutil.copytree(artifacts, bad)
    m = json.loads((bad / "manifest.json").read_text())
    m["feature_spec_hash"] = "0" * 64
    (bad / "manifest.json").write_text(json.dumps(m))
    with _real_client(monkeypatch, bad, np.zeros((6, 147), np.float32)) as c:
        r = c.get("/api/v1/health")
        assert r.status_code == 503 and r.json()["ready"] is False and r.json()["mode"] == "real"


def test_real_mode_without_task_files_is_unavailable_not_mock(artifacts, monkeypatch):
    with _real_client(monkeypatch, artifacts, None, with_extractor=False) as c:  # sin .task en el paquete
        r = c.get("/api/v1/health")
        assert r.status_code == 503 and r.json()["mode"] == "real"
