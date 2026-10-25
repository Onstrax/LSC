"""Mapeo uniforme de códigos de salida 0/2/3 de la CLI (design §19, §20).

Se ejercitan al menos ``acquire`` y ``validate``: ``0`` conforme, ``2`` config
inválida (``ConfigError``), ``3`` integridad (``IntegrityError``). La CLI no
importa cv2/mediapipe; usa imports perezosos por handler.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.lsc_data.cli import EXIT_CONFIG, EXIT_INTEGRITY, EXIT_OK, main
from src.lsc_data.packaging import write_bundle

FEATURE_SPEC_MIN = {
    "schema_version": "lsc.feature_spec.v1",
    "feature_version": "features_v1",
    "point_order": [f"p{i}" for i in range(49)],
    "coordinate_system": "shoulder_normalized_2d",
    "temporal_length": 6,
    "features_per_frame": 147,
    "dtype": "float32",
    "missing_policy": {},
    "quality_rules": {},
    "detector_versions": {},
    "model_asset_hashes": {},
    "normalization": {},
}


def _valid_bundle(out_dir):
    x = np.zeros((2, 6, 147), dtype=np.float32)
    write_bundle(
        out_dir,
        X=x,
        y=[0, 1],
        sample_id=["a", "b"],
        signer_id=["S000", "S001"],
        split=["train", "train"],
        quality_pass=[True, True],
        feature_spec=FEATURE_SPEC_MIN,
    )


# --- acquire ---------------------------------------------------------------
def test_acquire_malformed_config_exit_2(tmp_path):
    bad = tmp_path / "source.yaml"
    bad.write_text(": : : [bad", encoding="utf-8")
    assert main(["acquire", "--config", str(bad)]) == EXIT_CONFIG


def test_acquire_verify_only_exit_0():
    """acquire sin --archive sobre el source.yaml versionado -> 0 (sólo verifica)."""
    assert main(["acquire", "--config", "configs/data/source.yaml"]) == EXIT_OK


# --- validate --------------------------------------------------------------
def test_validate_missing_bundle_exit_2(tmp_path):
    assert main(["validate", "--bundle", str(tmp_path / "nope")]) == EXIT_CONFIG


def test_validate_valid_bundle_exit_0(tmp_path):
    out = tmp_path / "bundle"
    _valid_bundle(out)
    assert main(["validate", "--bundle", str(out)]) == EXIT_OK


def test_validate_integrity_failure_exit_3(tmp_path):
    """Un bundle con fuga de identidad entre splits -> exit 3 (integridad)."""
    out = tmp_path / "bundle"
    x = np.zeros((2, 6, 147), dtype=np.float32)
    write_bundle(
        out,
        X=x,
        y=[0, 1],
        sample_id=["a", "b"],
        signer_id=["S000", "S000"],  # mismo signer en dos splits -> fuga
        split=["train", "test"],
        quality_pass=[True, True],
        feature_spec=FEATURE_SPEC_MIN,
    )
    assert main(["validate", "--bundle", str(out)]) == EXIT_INTEGRITY


def test_validate_tampered_hash_exit_3(tmp_path):
    """feature_spec_hash divergente en el bundle -> exit 3 (integridad)."""
    out = tmp_path / "bundle"
    _valid_bundle(out)
    spec_path = out / "feature_spec.json"
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    spec["feature_spec_hash"] = "deadbeef"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    assert main(["validate", "--bundle", str(out)]) == EXIT_INTEGRITY
