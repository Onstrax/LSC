"""Detector inyectable y tipos de landmarks (design §8.2).

El paquete NO importa ``cv2`` ni ``mediapipe`` en el import. El extractor acepta
un detector que espeja la API IMAGE de MediaPipe Tasks. Para pruebas se usa
``FakeDetector`` (landmarks programados en memoria). ``RealDetector`` envuelve
MediaPipe Tasks (modo IMAGE); sus imports de MediaPipe son perezosos (dentro de
``__init__`` y de los ``detect_*``), nunca en el import del módulo.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np


# ---------------------------------------------------------------------------
# Tipos de resultado que imitan MediaPipe Tasks (modo IMAGE)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PoseLandmark:
    """Punto de pose normalizado 0..1.

    ``presence`` es opcional: su ausencia nunca invalida el punto; su presencia
    exige ``presence >= 0.5`` (regla ``pose_point_valid`` de §11).
    """

    x: float
    y: float
    visibility: float = 1.0
    presence: float | None = None


@dataclass(frozen=True)
class PoseResult:
    """Resultado de ``detect_pose`` (una lista de landmarks por persona).

    ``num_poses=2`` en el camino real; aquí ``pose_landmarks`` tiene longitud 0,
    1 o más según lo que devuelva el detector. Más de una persona ⇒
    ``MULTIPLE_PEOPLE`` (design §11).
    """

    pose_landmarks: list[list[PoseLandmark]] = field(default_factory=list)


@dataclass(frozen=True)
class HandLandmark:
    """Punto de mano normalizado 0..1.

    ``handedness`` (``"Left"``/``"Right"``) se IGNORA para la asociación: el lado
    anatómico lo fija la pose (design §11).
    """

    x: float
    y: float
    handedness: str | None = None


@dataclass(frozen=True)
class HandsResult:
    """Resultado de ``detect_hands`` (21 puntos por mano).

    ``hand_landmarks`` es una lista por mano detectada (``num_hands=2`` en el
    camino real). ``handedness`` se transporta pero no interviene en la
    asociación.
    """

    hand_landmarks: list[list[HandLandmark]] = field(default_factory=list)
    handedness: list[str] = field(default_factory=list)


@runtime_checkable
class DetectorProtocol(Protocol):
    """Interfaz del detector inyectable (espejo de MediaPipe Tasks, modo IMAGE)."""

    def detect_pose(self, image_rgb: np.ndarray) -> PoseResult: ...

    def detect_hands(self, image_rgb: np.ndarray) -> HandsResult: ...

    def close(self) -> None: ...


# ---------------------------------------------------------------------------
# FakeDetector — doble de pruebas
# ---------------------------------------------------------------------------
class FakeDetector:
    """Detector falso que devuelve landmarks programados por el test.

    Uso en pruebas (design §8.2): se construye con secuencias de resultados por
    cuadro, uno para pose y otro para manos. En cada llamada a ``detect_pose`` /
    ``detect_hands`` consume el siguiente elemento de su lista respectiva (en el
    orden temporal de los cuadros). Permite reproducir: cero personas, una
    persona, dos personas (MULTIPLE_PEOPLE), manos ambiguas, puntos no finitos,
    hombros ausentes, etc.

    Parámetros
    ----------
    pose_results:
        Secuencia de :class:`PoseResult`, uno por cuadro (en orden temporal).
    hands_results:
        Secuencia de :class:`HandsResult`, uno por cuadro (en orden temporal).

    Si una secuencia es más corta que el número de cuadros, el último elemento se
    reutiliza; si está vacía, se devuelve un resultado sin detecciones. No toca
    disco ni red; es puro en memoria.
    """

    def __init__(
        self,
        pose_results: list[PoseResult] | None = None,
        hands_results: list[HandsResult] | None = None,
    ) -> None:
        self._pose_results = list(pose_results or [])
        self._hands_results = list(hands_results or [])
        self._pose_idx = 0
        self._hands_idx = 0
        self.closed = False

    @staticmethod
    def _next(seq: list, idx: int, empty):
        if not seq:
            return empty, idx
        if idx < len(seq):
            return seq[idx], idx + 1
        # Agotada la lista: reutiliza el último elemento sin avanzar.
        return seq[-1], idx

    def detect_pose(self, image_rgb: np.ndarray) -> PoseResult:
        result, self._pose_idx = self._next(self._pose_results, self._pose_idx, PoseResult())
        return result

    def detect_hands(self, image_rgb: np.ndarray) -> HandsResult:
        result, self._hands_idx = self._next(self._hands_results, self._hands_idx, HandsResult())
        return result

    def close(self) -> None:
        self.closed = True


# ---------------------------------------------------------------------------
# RealDetector — envoltura real de MediaPipe Tasks (modo IMAGE)
# ---------------------------------------------------------------------------
class RealDetector:
    """Envoltura de MediaPipe Tasks (modo IMAGE).

    Carga los ``.task`` reales de ``pose_landmarker.task`` y
    ``hand_landmarker.task`` con ``running_mode=IMAGE``, ``num_poses=2``,
    ``num_hands=2`` y thresholds 0.5. Los imports de ``mediapipe`` son PEREZOSOS
    (dentro de ``__init__`` y de los ``detect_*``), nunca en el import del
    módulo (design §8.2/§16). Instanciarlo requiere ``mediapipe`` instalado y
    los artefactos reales.
    """

    def __init__(self, pose_task_path, hands_task_path) -> None:  # pragma: no cover - requiere mediapipe + artefactos
        # Import perezoso: nunca ocurre en el import del paquete.
        from mediapipe.tasks import python as mp_python  # type: ignore
        from mediapipe.tasks.python import vision as mp_vision  # type: ignore

        pose_opts = mp_vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(pose_task_path)),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_poses=2,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
        )
        hands_opts = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(hands_task_path)),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
        )
        self._pose = mp_vision.PoseLandmarker.create_from_options(pose_opts)
        self._hands = mp_vision.HandLandmarker.create_from_options(hands_opts)

    @staticmethod
    def _to_mp_image(image_rgb: np.ndarray):  # pragma: no cover - requiere mediapipe
        """Envuelve un arreglo RGB uint8 en un ``mp.Image`` (import perezoso)."""
        import mediapipe as mp  # type: ignore

        return mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

    def detect_pose(self, image_rgb: np.ndarray) -> PoseResult:  # pragma: no cover - requiere mediapipe + artefactos
        mp_image = self._to_mp_image(image_rgb)
        result = self._pose.detect(mp_image)
        if result.pose_landmarks is None:
            return PoseResult()
        return PoseResult(
            pose_landmarks=[
                [
                    PoseLandmark(
                        x=lm.x,
                        y=lm.y,
                        visibility=getattr(lm, "visibility", 1.0),
                        presence=getattr(lm, "presence", None),
                    )
                    for lm in persona
                ]
                for persona in result.pose_landmarks
            ]
        )

    def detect_hands(self, image_rgb: np.ndarray) -> HandsResult:  # pragma: no cover - requiere mediapipe + artefactos
        mp_image = self._to_mp_image(image_rgb)
        result = self._hands.detect(mp_image)
        if result.hand_landmarks is None:
            return HandsResult()
        handedness: list[str] = []
        for mano in getattr(result, "handedness", None) or []:
            if mano:
                handedness.append(mano[0].category_name)
        return HandsResult(
            hand_landmarks=[
                [HandLandmark(x=lm.x, y=lm.y) for lm in mano]
                for mano in result.hand_landmarks
            ],
            handedness=handedness,
        )

    def close(self) -> None:  # pragma: no cover - requiere mediapipe + artefactos
        self._pose.close()
        self._hands.close()
