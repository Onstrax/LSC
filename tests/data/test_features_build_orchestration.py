"""Orquestación de ``features`` (DO07/FEAT-003): construcción real del bundle.

Ejercita ``run_features`` con un ``FakeDetector`` inyectado y un ``frame_loader``
sintético (6 arreglos RGB uint8 en memoria), sobre un manifiesto + splits
mínimos, SIN corpus ni ``.task`` reales. Verifica que el bundle resultante:

- se escribe con ``features.npz`` + ``feature_spec.json`` + ``exclusions.csv``,
- pasa ``validate`` (exit 0),
- tiene la columna ``split`` no vacía y coherente con ``splits.json``,
- tiene ``X`` con forma ``(N, 6, 147)`` float32.
"""

from __future__ import annotations

import json

import numpy as np

from contracts.canonical import sample_id as canon_sample_id
from contracts.canonical import sequence_sha256 as canon_sequence_sha256
from src.lsc_data.features_build import run_features
from src.lsc_data.validate import validate
from src.lsc_data.packaging import read_features_npz
from src.lsc_features.detector import (
    FakeDetector,
    HandLandmark,
    HandsResult,
    PoseLandmark,
    PoseResult,
)

SOURCE_VERSION = "SYNTHETIC-v0"
WIDTH = 640
HEIGHT = 480

# Geometría válida (misma convención que tests/features/conftest.py).
LEFT_SHOULDER_XY = (0.375, 0.40)
RIGHT_SHOULDER_XY = (0.625, 0.40)
LEFT_WRIST_XY = (0.35, 0.60)
RIGHT_WRIST_XY = (0.65, 0.60)
NOSE_XY = (0.50, 0.30)


def _make_pose() -> list[PoseLandmark]:
    pts = [PoseLandmark(NOSE_XY[0], NOSE_XY[1])]
    for _ in range(1, 11):
        pts.append(PoseLandmark(0.5, 0.5))
    pts.append(PoseLandmark(*LEFT_SHOULDER_XY))   # 11
    pts.append(PoseLandmark(*RIGHT_SHOULDER_XY))  # 12
    pts.append(PoseLandmark(0.4, 0.5))            # 13
    pts.append(PoseLandmark(0.6, 0.5))            # 14
    pts.append(PoseLandmark(*LEFT_WRIST_XY))      # 15
    pts.append(PoseLandmark(*RIGHT_WRIST_XY))     # 16
    return pts


def _make_hand(center) -> list[HandLandmark]:
    cx, cy = center
    pts = [HandLandmark(cx, cy)]
    for i in range(1, 21):
        pts.append(HandLandmark(cx + 0.02 * (i % 3 - 1), cy + 0.02 * (i % 2)))
    return pts


def _good_detector() -> FakeDetector:
    pose = [PoseResult(pose_landmarks=[_make_pose()]) for _ in range(6)]
    hands = [
        HandsResult(hand_landmarks=[_make_hand(LEFT_WRIST_XY), _make_hand(RIGHT_WRIST_XY)])
        for _ in range(6)
    ]
    return FakeDetector(pose_results=pose, hands_results=hands)


def _record(signer_id: str, class_id: int) -> dict:
    shas = ["%064x" % (abs(hash(f"{signer_id}-{class_id}|{i}")) % (16 ** 64)) for i in range(6)]
    seq = canon_sequence_sha256(shas)
    sid = canon_sample_id(
        source_version=SOURCE_VERSION,
        signer_id=signer_id,
        class_id=class_id,
        repetition_id=0,
        frame_sha256=shas,
    )
    return {
        "schema_version": "lsc.manifest.v1",
        "sample_id": sid,
        "source": "SYNTHETIC",
        "source_version": SOURCE_VERSION,
        "signer_id": signer_id,
        "class_id": class_id,
        "raw_label": str(class_id),
        "repetition_id": 0,
        "frame_paths": [f"{signer_id}/{class_id}/frame_{i}.jpg" for i in range(6)],
        "frame_sha256": shas,
        "frame_indices": [0, 1, 2, 3, 4, 5],
        "width": WIDTH,
        "height": HEIGHT,
        "n_frames": 6,
        "sequence_sha256": seq,
        "structural_status": "valid",
    }


def _write_manifest(path, records: list[dict]) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False))
            fh.write("\n")


def _synthetic_frames(_record_unused) -> list[np.ndarray]:
    return [np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8) for _ in range(6)]


def test_run_features_builds_valid_bundle_with_split(tmp_path):
    """run_features con detector/loader inyectados produce un bundle válido."""
    # Dos firmantes distintos en train/test, clases 0 y 1.
    records = [
        _record("S000", 0),
        _record("S000", 1),
        _record("S001", 0),
        _record("S001", 1),
    ]
    manifest = tmp_path / "manifest.jsonl"
    _write_manifest(manifest, records)

    split_map = {
        "train": [records[0]["sample_id"], records[1]["sample_id"]],
        "val": [],
        "test": [records[2]["sample_id"], records[3]["sample_id"]],
    }
    splits = tmp_path / "splits.json"
    splits.write_text(
        json.dumps({"sample_ids": split_map}, ensure_ascii=False),
        encoding="utf-8",
    )

    out_dir = tmp_path / "bundle"
    code = run_features(
        config_path="configs/data/features_v1.yaml",
        manifest_path=manifest,
        splits_path=splits,
        out_dir=out_dir,
        resume=False,
        detector=_good_detector(),
        frame_loader=_synthetic_frames,
    )
    assert code == 0

    # Artefactos presentes.
    assert (out_dir / "features.npz").is_file()
    assert (out_dir / "feature_spec.json").is_file()
    assert (out_dir / "exclusions.csv").is_file()

    # validate conforme (exit 0).
    assert validate(out_dir) == 0

    # Columna split no vacía y coherente.
    bundle = read_features_npz(out_dir / "features.npz")
    assert bundle.X.shape == (4, 6, 147)
    assert bundle.X.dtype == np.float32
    splits_present = set(bundle.split.tolist())
    assert "" not in splits_present
    assert splits_present == {"train", "test"}

    id_to_split = dict(zip(bundle.sample_id.tolist(), bundle.split.tolist()))
    assert id_to_split[records[0]["sample_id"]] == "train"
    assert id_to_split[records[2]["sample_id"]] == "test"
