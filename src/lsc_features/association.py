"""Asociación mano-pose de costo mínimo (design §11).

Para cada muñeca detectada por el Hand Landmarker, el costo frente a una muñeca
de pose es la distancia en px muñeca-hand vs muñeca-pose, normalizada por la
distancia de hombros DEL PROPIO CUADRO. El lado anatómico lo fija la pose:
pose-15 = mano IZQUIERDA de la persona, pose-16 = mano DERECHA. El ``handedness``
de MediaPipe se IGNORA.

Precondición de hombros por cuadro: si el cuadro no tiene hombros válidos, NO se
asocia ninguna mano, el cuadro contribuye 0 a ``frames_with_hand`` y NO emite
``HAND_ASSOCIATION_AMBIGUOUS`` (esta función no se llama para esos cuadros).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .geometry import LEFT_WRIST, RIGHT_WRIST, is_finite_xy, pose_point_valid, to_px

# Índice de la muñeca dentro de los 21 landmarks de una mano (MediaPipe: 0).
HAND_WRIST_INDEX = 0


@dataclass
class AssociationResult:
    """Resultado de asociar manos a muñecas de pose en un cuadro.

    ``left_hand`` / ``right_hand`` son la lista de 21 ``HandLandmark`` asignada a
    cada lado anatómico, o ``None`` si ese lado quedó ausente. ``ambiguous`` marca
    ``HAND_ASSOCIATION_AMBIGUOUS`` (sólo posible con 2 manos y gap pequeño).
    """

    left_hand: list | None = None
    right_hand: list | None = None
    ambiguous: bool = False

    @property
    def has_any_hand(self) -> bool:
        return self.left_hand is not None or self.right_hand is not None


def _wrist_px(hand_landmarks, width, height) -> tuple[float, float] | None:
    if hand_landmarks is None or len(hand_landmarks) <= HAND_WRIST_INDEX:
        return None
    wrist = hand_landmarks[HAND_WRIST_INDEX]
    x = float(wrist.x)
    y = float(wrist.y)
    if not is_finite_xy(x, y):
        return None
    return to_px(x, y, width, height)


def _pose_wrist_px(pose_landmarks, index, width, height, valid_kwargs) -> tuple[float, float] | None:
    if pose_landmarks is None or len(pose_landmarks) <= index:
        return None
    lm = pose_landmarks[index]
    if not pose_point_valid(lm, **valid_kwargs):
        return None
    return to_px(float(lm.x), float(lm.y), width, height)


def _cost(hand_wrist, pose_wrist, shoulder_distance) -> float:
    dist_px = math.hypot(hand_wrist[0] - pose_wrist[0], hand_wrist[1] - pose_wrist[1])
    return dist_px / shoulder_distance


def associate_hands(
    pose_landmarks: list,
    hands_landmarks: list,
    width: int,
    height: int,
    shoulder_distance: float,
    *,
    cost_max: float,
    ambiguous_gap: float,
    visibility_threshold: float,
    presence_threshold: float,
    use_presence: bool,
) -> AssociationResult:
    """Asocia manos a los lados anatómicos de la pose (design §11).

    Precondición: ``shoulder_distance`` corresponde a un cuadro con hombros
    válidos (quien llama ya lo verificó). ``hands_landmarks`` es la lista de manos
    detectadas (cada una, 21 landmarks). Ramas 0/1/2 manos según §11.
    """
    valid_kwargs = dict(
        visibility_threshold=visibility_threshold,
        presence_threshold=presence_threshold,
        use_presence=use_presence,
    )
    pose_left = _pose_wrist_px(pose_landmarks, LEFT_WRIST, width, height, valid_kwargs)
    pose_right = _pose_wrist_px(pose_landmarks, RIGHT_WRIST, width, height, valid_kwargs)

    # Muñecas de mano detectadas con coordenadas finitas.
    hands = []
    for hand in hands_landmarks or []:
        wrist = _wrist_px(hand, width, height)
        if wrist is not None:
            hands.append((hand, wrist))

    result = AssociationResult()
    n = len(hands)

    if n == 0:
        return result

    if n == 1:
        hand, wrist = hands[0]
        candidates: list[tuple[float, str]] = []
        if pose_left is not None:
            candidates.append((_cost(wrist, pose_left, shoulder_distance), "left"))
        if pose_right is not None:
            candidates.append((_cost(wrist, pose_right, shoulder_distance), "right"))
        if not candidates:
            return result
        candidates.sort(key=lambda c: c[0])
        best_cost, best_side = candidates[0]
        if best_cost <= cost_max:
            if best_side == "left":
                result.left_hand = hand
            else:
                result.right_hand = hand
        # Gap indefinido con una sola mano: NUNCA ambiguo.
        return result

    # n >= 2: usar las dos primeras muñecas; enumerar asignaciones inyectivas.
    (hand_a, wrist_a), (hand_b, wrist_b) = hands[0], hands[1]

    assignments = []  # (cost_total, {"left": hand|None, "right": hand|None})
    # Asignación 1: A->izq, B->der.
    if pose_left is not None and pose_right is not None:
        c_al = _cost(wrist_a, pose_left, shoulder_distance)
        c_br = _cost(wrist_b, pose_right, shoulder_distance)
        if c_al <= cost_max and c_br <= cost_max:
            assignments.append((c_al + c_br, {"left": hand_a, "right": hand_b}))
    # Asignación 2: A->der, B->izq.
    if pose_left is not None and pose_right is not None:
        c_ar = _cost(wrist_a, pose_right, shoulder_distance)
        c_bl = _cost(wrist_b, pose_left, shoulder_distance)
        if c_ar <= cost_max and c_bl <= cost_max:
            assignments.append((c_ar + c_bl, {"left": hand_b, "right": hand_a}))

    if not assignments:
        return result

    assignments.sort(key=lambda a: a[0])

    if len(assignments) == 1:
        # Una sola asignación completa: se usa sin regla de gap.
        mapping = assignments[0][1]
        result.left_hand = mapping["left"]
        result.right_hand = mapping["right"]
        return result

    # Dos asignaciones completas: aplicar regla de gap.
    gap = assignments[1][0] - assignments[0][0]
    if gap < ambiguous_gap:
        result.ambiguous = True  # ambas manos implicadas quedan ausentes
        return result
    mapping = assignments[0][1]
    result.left_hand = mapping["left"]
    result.right_hand = mapping["right"]
    return result
