"""Fixtures del extractor: FakeDetector y geometría sintética (design §8.2, §20).

Construye landmarks de pose/mano programados EN MEMORIA (coordenadas
normalizadas 0..1) para ejercitar el extractor sin ``cv2``/``mediapipe`` ni
imágenes reales. El extractor convierte a px con ``x*W``, ``y*H``.

Convención de pose (MediaPipe): índice 0 = nariz, 11/12 = hombros izq/der,
15/16 = muñecas izq/der (lado anatómico; design §11). Se rellenan los índices
0..16 para que ``pose_landmarks[16]`` exista.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.lsc_features.detector import (
    FakeDetector,
    HandLandmark,
    HandsResult,
    PoseLandmark,
    PoseResult,
)

WIDTH = 640
HEIGHT = 480

# Posiciones normalizadas por defecto (fracción de W/H).
# Hombros separados ~0.25 de ancho (>= 0.08*W), centrados.
LEFT_SHOULDER_XY = (0.375, 0.40)   # pose 11 (mano IZQ anatómica)
RIGHT_SHOULDER_XY = (0.625, 0.40)  # pose 12 (mano DER anatómica)
LEFT_WRIST_XY = (0.35, 0.60)       # pose 15
RIGHT_WRIST_XY = (0.65, 0.60)      # pose 16
NOSE_XY = (0.50, 0.30)             # pose 0

REPO_ROOT = Path(__file__).resolve().parents[2]
FEATURE_SPEC = json.loads(
    (REPO_ROOT / "contracts" / "feature_spec.json").read_text(encoding="utf-8")
)
MODEL_PATHS = {"pose": "pose_landmarker.task", "hands": "hand_landmarker.task"}


def make_pose(
    *,
    left_shoulder=LEFT_SHOULDER_XY,
    right_shoulder=RIGHT_SHOULDER_XY,
    left_wrist=LEFT_WRIST_XY,
    right_wrist=RIGHT_WRIST_XY,
    nose=NOSE_XY,
    visibility: float = 1.0,
    presence=None,
) -> list[PoseLandmark]:
    """Lista de 17 landmarks de pose (índices 0..16) con los puntos clave."""
    pts = [PoseLandmark(nose[0], nose[1], visibility, presence)]
    # 1..10: rellenos válidos (no se usan, pero deben existir).
    for _ in range(1, 11):
        pts.append(PoseLandmark(0.5, 0.5, visibility, presence))
    pts.append(PoseLandmark(left_shoulder[0], left_shoulder[1], visibility, presence))   # 11
    pts.append(PoseLandmark(right_shoulder[0], right_shoulder[1], visibility, presence))  # 12
    pts.append(PoseLandmark(0.4, 0.5, visibility, presence))  # 13
    pts.append(PoseLandmark(0.6, 0.5, visibility, presence))  # 14
    pts.append(PoseLandmark(left_wrist[0], left_wrist[1], visibility, presence))   # 15
    pts.append(PoseLandmark(right_wrist[0], right_wrist[1], visibility, presence))  # 16
    return pts


def make_pose_result(pose=None, *, n_people: int = 1) -> PoseResult:
    """``PoseResult`` con ``n_people`` poses (0 -> sin persona)."""
    if n_people == 0:
        return PoseResult(pose_landmarks=[])
    base = pose if pose is not None else make_pose()
    return PoseResult(pose_landmarks=[base for _ in range(n_people)])


def make_hand(center, *, spread: float = 0.02) -> list[HandLandmark]:
    """21 landmarks de una mano; el landmark 0 (muñeca) en ``center``."""
    cx, cy = center
    pts = [HandLandmark(cx, cy)]  # 0 = muñeca
    for i in range(1, 21):
        pts.append(HandLandmark(cx + spread * (i % 3 - 1), cy + spread * (i % 2)))
    return pts


def make_hands_result(*centers) -> HandsResult:
    """``HandsResult`` con una mano por centro de muñeca dado."""
    return HandsResult(hand_landmarks=[make_hand(c) for c in centers])


def make_detector(pose_results, hands_results) -> FakeDetector:
    return FakeDetector(pose_results=pose_results, hands_results=hands_results)


def good_frame_pose() -> PoseResult:
    return make_pose_result(n_people=1)


def good_frame_hands() -> HandsResult:
    # Dos manos sobre las muñecas de pose (asignación no ambigua).
    return make_hands_result(LEFT_WRIST_XY, RIGHT_WRIST_XY)


@pytest.fixture
def feature_spec():
    return dict(FEATURE_SPEC)


@pytest.fixture
def model_paths():
    return dict(MODEL_PATHS)


@pytest.fixture
def frames_rgb():
    """Seis cuadros RGB uint8 del mismo tamaño (contenido irrelevante: FakeDetector)."""
    return [np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8) for _ in range(6)]


@pytest.fixture
def builders():
    """Expone los constructores de geometría a las pruebas."""
    return {
        "make_pose": make_pose,
        "make_pose_result": make_pose_result,
        "make_hand": make_hand,
        "make_hands_result": make_hands_result,
        "make_detector": make_detector,
        "good_frame_pose": good_frame_pose,
        "good_frame_hands": good_frame_hands,
        "WIDTH": WIDTH,
        "HEIGHT": HEIGHT,
        "LEFT_WRIST_XY": LEFT_WRIST_XY,
        "RIGHT_WRIST_XY": RIGHT_WRIST_XY,
    }
