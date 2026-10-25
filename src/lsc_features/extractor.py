"""FeatureExtractor: extractor inyectable y orquestación pura (design §8.3, §9-§13).

Firma EXACTA (fijada contra ``api/adapters/real.py``, sólo lectura)::

    FeatureExtractor(config, model_paths, detector=None)

``config`` acepta un ``dict`` de ``feature_spec`` (no exige pydantic).
``model_paths`` DEBE tener EXACTAMENTE las claves ``{"pose", "hands"}``; una clave
faltante o sobrante produce ``CONFIG_MISMATCH`` en ``init()`` sin tocar disco. Si
se inyecta ``detector``, se usa tal cual y ``model_paths`` NO se abre ni hashea.
Si ``detector`` es ``None``, ``init()`` verifica existencia+hash SOLO de ``"pose"``
y ``"hands"`` (camino real NO ejercitado, bloque DO05). Nada se carga en import.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from contracts.canonical import feature_spec_hash as _feature_spec_hash

from . import geometry as geo
from . import reason_codes as rc
from .association import associate_hands
from .detector import DetectorProtocol
from .geometry import POINTS_PER_INSTANT, POSE_INDICES
from .quality import QualityCounters, evaluate_quality
from .types import FeatureResult

FRAME_COUNT = 6
FEATURE_DIM = 147
SEQUENCE_SHAPE = (FRAME_COUNT, FEATURE_DIM)
REQUIRED_MODEL_PATH_KEYS = frozenset({"pose", "hands"})

# Posiciones de los 49 puntos dentro del triple-vector (x,y,m):
#   0..6  pose (0,11,12,13,14,15,16)
#   7..27 mano izquierda (lh_0..lh_20)
#   28..48 mano derecha (rh_0..rh_20)
_POSE_SLOT = {pose_index: slot for slot, pose_index in enumerate(POSE_INDICES)}
_LEFT_HAND_OFFSET = len(POSE_INDICES)       # 7
_RIGHT_HAND_OFFSET = len(POSE_INDICES) + 21  # 28


class ConfigMismatchError(ValueError):
    """``model_paths`` con claves inválidas o config incompatible (CONFIG_MISMATCH)."""


class FeatureExtractor:
    """Extractor de características con detector inyectable (design §8.3)."""

    def __init__(
        self,
        config: Mapping[str, Any],
        model_paths: Mapping[str, str | Path],
        detector: DetectorProtocol | None = None,
    ) -> None:
        self._config = dict(config)
        self._model_paths = dict(model_paths)
        self._detector = detector
        self._initialized = False

        # Hash del feature_spec cargado (para comparar con el del modelo).
        self.feature_spec_hash = self._resolve_feature_spec_hash(self._config)

        # Reglas de calidad desde config (con defaults del feature_spec §4.2).
        rules = dict(self._config.get("quality_rules", {}))
        self._visibility_threshold = float(rules.get("visibility_threshold", 0.5))
        self._presence_threshold = float(rules.get("presence_threshold", 0.5))
        self._use_presence = bool(rules.get("use_presence", False))
        self._min_shoulder_frac = float(rules.get("min_shoulder_distance_frac_width", 0.08))
        self._min_valid_shoulder_frames = int(rules.get("min_valid_shoulder_frames", 5))
        self._min_hand_frames = int(rules.get("min_hand_frames", 4))
        self._max_people = int(rules.get("max_people", 1))
        self._geometry_outlier_abs = float(rules.get("geometry_outlier_abs", 10.0))
        self._cost_max = float(rules.get("hand_association_cost_max", 0.6))
        self._ambiguous_gap = float(rules.get("hand_association_ambiguous_gap", 0.05))

        # init() explícita: valida model_paths siempre; toca disco sólo sin detector.
        self.init()

    @staticmethod
    def _resolve_feature_spec_hash(config: Mapping[str, Any]) -> str:
        """Hash del feature_spec: usa el almacenado si existe, si no lo recomputa."""
        stored = config.get("feature_spec_hash")
        if stored:
            return str(stored)
        try:
            return _feature_spec_hash(config)
        except Exception:
            return ""

    def init(self) -> None:
        """Verificación explícita (design §8.3). NO ocurre en el import.

        - ``model_paths`` DEBE tener EXACTAMENTE ``{"pose", "hands"}``; faltante o
          sobrante ⇒ ``CONFIG_MISMATCH`` (``ConfigMismatchError``) sin tocar disco.
        - Con ``detector`` inyectado: ``model_paths`` no se abre ni hashea.
        - Con ``detector=None``: se verifica existencia (+hash, bloque DO05) SOLO
          de ``"pose"`` y ``"hands"``.
        """
        keys = set(self._model_paths)
        if keys != REQUIRED_MODEL_PATH_KEYS:
            raise ConfigMismatchError(
                f"model_paths debe tener exactamente {sorted(REQUIRED_MODEL_PATH_KEYS)}; "
                f"recibido {sorted(keys)} [{rc.CONFIG_MISMATCH}]"
            )

        if self._detector is None:
            # Camino real NO ejercitado (bloque DO05): verifica existencia de .task.
            for key in ("pose", "hands"):  # pragma: no cover - DO05
                path = Path(self._model_paths[key])
                if not path.exists():
                    raise ConfigMismatchError(
                        f"No se encontró el artefacto '{key}' en {path}; el extractor "
                        f"no puede inicializar el detector real. [{rc.CONFIG_MISMATCH}]"
                    )
            # La verificación de hash contra detector_versions/model_asset_hashes
            # y la carga del RealDetector se desbloquean en DO05.

        self._initialized = True

    # ------------------------------------------------------------------
    # Extracción
    # ------------------------------------------------------------------
    def extract_sequence(self, frames_rgb: list[np.ndarray]) -> FeatureResult:
        """Procesa 6 cuadros RGB uint8 y devuelve un :class:`FeatureResult`.

        Invariante de salida (§8.1): ``x`` siempre es ``np.zeros((6,147),
        float32)`` con sólo los triples válidos escritos, aun con
        ``quality_pass=False``.
        """
        start = time.perf_counter()
        x = np.zeros(SEQUENCE_SHAPE, dtype=np.float32)
        counters = QualityCounters()

        counters.frame_count = len(frames_rgb) if frames_rgb is not None else 0

        # --- Validación estructural de la entrada (§18) ---
        if counters.frame_count != FRAME_COUNT:
            counters.invalid_frame_count = True
            return self._finish(x, counters, start)

        ref_shape = None
        structural_fail = False
        for frame in frames_rgb:
            if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
                counters.invalid_image = True
                structural_fail = True
                break
            if ref_shape is None:
                ref_shape = frame.shape
            elif frame.shape != ref_shape:
                counters.size_mismatch = True
                structural_fail = True
                break
        if structural_fail:
            return self._finish(x, counters, start)

        height, width = ref_shape[0], ref_shape[1]

        # --- Detección por cuadro (§9-§11) ---
        per_frame = []  # (pose_landmarks|None, hands_landmarks)
        for frame in frames_rgb:
            pose_result = self._detector.detect_pose(frame)
            hands_result = self._detector.detect_hands(frame)
            poses = list(getattr(pose_result, "pose_landmarks", []) or [])
            if len(poses) > self._max_people:
                counters.multiple_people = True
            pose_landmarks = poses[0] if len(poses) >= 1 else None
            hands_landmarks = list(getattr(hands_result, "hand_landmarks", []) or [])
            per_frame.append((pose_landmarks, hands_landmarks))

        if counters.multiple_people:
            return self._finish(x, counters, start)

        # --- Hombros por cuadro: punto medio y distancia (§12.2) ---
        shoulder_info: list[tuple[tuple[float, float], float] | None] = []
        midpoints: list[tuple[float, float]] = []
        distances: list[float] = []
        for pose_landmarks, _ in per_frame:
            info = geo.shoulder_midpoint_and_distance(
                pose_landmarks,
                width,
                height,
                min_shoulder_distance_frac_width=self._min_shoulder_frac,
                visibility_threshold=self._visibility_threshold,
                presence_threshold=self._presence_threshold,
                use_presence=self._use_presence,
            )
            shoulder_info.append(info)
            if info is not None:
                midpoints.append(info[0])
                distances.append(info[1])

        counters.valid_shoulder_frames = len(midpoints)

        center_scale = geo.sequence_center_scale(midpoints, distances)

        valid_kwargs = dict(
            visibility_threshold=self._visibility_threshold,
            presence_threshold=self._presence_threshold,
            use_presence=self._use_presence,
        )

        # --- Construcción de triples por cuadro (§11-§12) ---
        if center_scale is not None:
            center, scale = center_scale
            for f_idx, (pose_landmarks, hands_landmarks) in enumerate(per_frame):
                info = shoulder_info[f_idx]
                # Puntos de pose conservados.
                if pose_landmarks is not None:
                    for pose_index, slot in _POSE_SLOT.items():
                        if pose_index < len(pose_landmarks):
                            lm = pose_landmarks[pose_index]
                            if geo.pose_point_valid(lm, **valid_kwargs):
                                px, py = geo.to_px(float(lm.x), float(lm.y), width, height)
                                nx, ny = geo.normalize_point(px, py, center, scale)
                                if self._write_triple(x, f_idx, slot, nx, ny, counters):
                                    return self._finish(x, counters, start)

                # Asociación de manos sólo en cuadros con hombros válidos (§11).
                if info is None:
                    continue
                shoulder_distance = info[1]
                assoc = associate_hands(
                    pose_landmarks,
                    hands_landmarks,
                    width,
                    height,
                    shoulder_distance,
                    cost_max=self._cost_max,
                    ambiguous_gap=self._ambiguous_gap,
                    **valid_kwargs,
                )
                if assoc.ambiguous:
                    counters.hand_association_ambiguous = True
                if assoc.has_any_hand:
                    counters.frames_with_hand += 1
                if assoc.left_hand is not None:
                    if self._write_hand(x, f_idx, _LEFT_HAND_OFFSET, assoc.left_hand, width, height, center, scale, counters):
                        return self._finish(x, counters, start)
                if assoc.right_hand is not None:
                    if self._write_hand(x, f_idx, _RIGHT_HAND_OFFSET, assoc.right_hand, width, height, center, scale, counters):
                        return self._finish(x, counters, start)

        return self._finish(x, counters, start)

    # ------------------------------------------------------------------
    # Helpers de escritura (mantienen la invariante de forma/dtype)
    # ------------------------------------------------------------------
    def _write_triple(self, x, f_idx, slot, nx, ny, counters) -> bool:
        """Escribe un triple (x,y,m=1). Devuelve ``True`` si hay GEOMETRY_OUTLIER."""
        if abs(nx) > self._geometry_outlier_abs or abs(ny) > self._geometry_outlier_abs:
            counters.geometry_outlier = True
            return True
        base = slot * 3
        x[f_idx, base] = np.float32(nx)
        x[f_idx, base + 1] = np.float32(ny)
        x[f_idx, base + 2] = np.float32(1.0)
        return False

    def _write_hand(self, x, f_idx, offset, hand_landmarks, width, height, center, scale, counters) -> bool:
        """Escribe los 21 puntos de una mano asociada. ``True`` si GEOMETRY_OUTLIER."""
        for i in range(21):
            if i >= len(hand_landmarks):
                continue
            lm = hand_landmarks[i]
            x_norm_raw = float(lm.x)
            y_norm_raw = float(lm.y)
            if not geo.is_finite_xy(x_norm_raw, y_norm_raw):
                continue  # punto no finito -> ausente (m=0)
            px, py = geo.to_px(x_norm_raw, y_norm_raw, width, height)
            nx, ny = geo.normalize_point(px, py, center, scale)
            if self._write_triple(x, f_idx, offset + i, nx, ny, counters):
                return True
        return False

    def _finish(self, x: np.ndarray, counters: QualityCounters, start: float) -> FeatureResult:
        """Evalúa calidad, mide latencia y devuelve el FeatureResult (invariante §8.1)."""
        # Garantizar la invariante de forma/dtype pase lo que pase.
        if x.shape != SEQUENCE_SHAPE or x.dtype != np.float32:
            x = np.asarray(x, dtype=np.float32).reshape(SEQUENCE_SHAPE)
        quality_pass, codes = evaluate_quality(
            counters,
            min_valid_shoulder_frames=self._min_valid_shoulder_frames,
            min_hand_frames=self._min_hand_frames,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return FeatureResult(
            x=x,
            quality_pass=quality_pass,
            reason_codes=codes,
            valid_shoulders=int(counters.valid_shoulder_frames),
            frames_with_hand=int(counters.frames_with_hand),
            feature_spec_hash=self.feature_spec_hash,
            elapsed_ms=float(elapsed_ms),
        )
