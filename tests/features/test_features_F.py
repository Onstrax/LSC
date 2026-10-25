"""Pruebas del extractor F01..F12 (design §11-§15, §20).

Geometría sintética + FakeDetector; nunca red, corpus ni ``.task``. Las de
fallo deliberado (F08d, F10, F11, F12) deben ponerse rojas si se neutraliza la
detección del defecto.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.lsc_features import FeatureExtractor, augment_train
from src.lsc_features.temporal import sample_indices

from .conftest import (
    FEATURE_SPEC,
    HEIGHT,
    LEFT_WRIST_XY,
    MODEL_PATHS,
    RIGHT_WRIST_XY,
    WIDTH,
    good_frame_hands,
    good_frame_pose,
    make_detector,
    make_hands_result,
    make_pose,
    make_pose_result,
)


def _frames():
    return [np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8) for _ in range(6)]


def _extract(pose_results, hands_results, frames=None):
    det = make_detector(pose_results, hands_results)
    ex = FeatureExtractor(dict(FEATURE_SPEC), dict(MODEL_PATHS), detector=det)
    return ex.extract_sequence(frames if frames is not None else _frames())


# --- F01 -------------------------------------------------------------------
def test_F01_one_person_both_hands_quality_pass():
    r = _extract([good_frame_pose()] * 6, [good_frame_hands()] * 6)
    assert r.x.shape == (6, 147)
    assert r.quality_pass is True
    assert r.reason_codes == []
    assert r.valid_shoulders == 6
    assert r.frames_with_hand == 6
    # Suma de máscaras coherente: >0 en cada cuadro.
    masks = r.x[:, 2::3]
    assert masks.sum() > 0


# --- F02 -------------------------------------------------------------------
def test_F02_shape_dtype_and_triple_order():
    # Camino con quality_pass=False igualmente mantiene forma/dtype (invariante §8.1).
    r_fail = _extract([make_pose_result(n_people=0)] * 6, [good_frame_hands()] * 6)
    assert r_fail.x.shape == (6, 147)
    assert r_fail.x.dtype == np.float32

    r_ok = _extract([good_frame_pose()] * 6, [good_frame_hands()] * 6)
    assert r_ok.x.shape == (6, 147)
    assert r_ok.x.dtype == np.float32
    # Orden (x, y, m): el tercer canal es máscara {0,1}.
    m = r_ok.x[:, 2::3]
    assert set(np.unique(m)).issubset({0.0, 1.0})


# --- F03 -------------------------------------------------------------------
def test_F03_normalization_common_center_scale():
    """Centro/escala medianos comunes: la nariz cae en el lugar esperado."""
    r = _extract([good_frame_pose()] * 6, [good_frame_hands()] * 6)
    # nose(0.5,0.3): center=(320,192) px, scale=160 px -> (0, -0.3).
    assert r.x[0, 0] == pytest.approx(0.0, abs=1e-5)
    assert r.x[0, 1] == pytest.approx(-0.3, abs=1e-5)
    # pose15 muñeca izq (0.35,0.60) -> (-0.6, 0.6).
    assert r.x[0, 15] == pytest.approx(-0.6, abs=1e-5)
    assert r.x[0, 16] == pytest.approx(0.6, abs=1e-5)


# --- F04 -------------------------------------------------------------------
def test_F04_absent_point_is_zero_and_real_zero_preserved():
    """m=0 -> x=y=0; una coordenada cero real con m=1 se conserva."""
    # Nariz en el centro de la secuencia -> normalizado (0,0) con m=1 (cero real).
    pose = make_pose(nose=(0.5, 0.4))  # px (320,192) = center -> (0,0)
    r = _extract([make_pose_result(pose)] * 6, [good_frame_hands()] * 6)
    assert r.x[0, 0] == pytest.approx(0.0)
    assert r.x[0, 1] == pytest.approx(0.0)
    assert r.x[0, 2] == 1.0  # cero REAL: la máscara sigue en 1

    # Un punto ausente (sin hombros -> sin manos): triple todo cero con m=0.
    r2 = _extract([make_pose_result(n_people=0)] * 6, [good_frame_hands()] * 6)
    # Sin pose válida no se escribe nada: todos los triples en cero, m=0.
    assert np.all(r2.x == 0.0)


# --- F05 -------------------------------------------------------------------
def test_F05_geometry_outlier_rejected():
    """Un valor normalizado |v|>10 -> GEOMETRY_OUTLIER, quality_pass=False."""
    pose = make_pose(nose=(20.0, 0.3))  # px enorme -> |x_norm| >> 10
    r = _extract([make_pose_result(pose)] * 6, [good_frame_hands()] * 6)
    assert "GEOMETRY_OUTLIER" in r.reason_codes
    assert r.quality_pass is False


# --- F06 -------------------------------------------------------------------
def test_F06_multiple_people_rejected():
    r = _extract([make_pose_result(n_people=2)] * 6, [good_frame_hands()] * 6)
    assert "MULTIPLE_PEOPLE" in r.reason_codes
    assert r.quality_pass is False


# --- F07 -------------------------------------------------------------------
def test_F07_insufficient_pose():
    """<5 cuadros con hombros válidos -> INSUFFICIENT_POSE, quality_pass=False."""
    good = good_frame_pose()
    none = make_pose_result(n_people=0)
    # Sólo 4 cuadros con pose válida.
    poses = [good, good, good, good, none, none]
    r = _extract(poses, [good_frame_hands()] * 6)
    assert r.valid_shoulders == 4
    assert "INSUFFICIENT_POSE" in r.reason_codes
    assert r.quality_pass is False


# --- F08a ------------------------------------------------------------------
def test_F08a_zero_hands_no_association():
    """0 manos en todos los cuadros -> frames_with_hand=0, sin ambigüedad."""
    r = _extract([good_frame_pose()] * 6, [make_hands_result()] * 6)
    assert r.frames_with_hand == 0
    assert "HAND_ASSOCIATION_AMBIGUOUS" not in r.reason_codes
    assert "INSUFFICIENT_HANDS" in r.reason_codes  # <4 cuadros con mano


# --- F08b ------------------------------------------------------------------
def test_F08b_one_hand_cost_threshold():
    """1 mano: costo<=0.6 asigna; costo>0.6 queda ausente; nunca ambigua."""
    # Mano justo sobre la muñeca izquierda -> costo ~0 (<=0.6): se asigna.
    close = make_hands_result(LEFT_WRIST_XY)
    r_close = _extract([good_frame_pose()] * 6, [close] * 6)
    assert r_close.frames_with_hand == 6
    assert "HAND_ASSOCIATION_AMBIGUOUS" not in r_close.reason_codes

    # Mano lejísimos de ambas muñecas de pose -> costo>0.6: ausente.
    far = make_hands_result((0.02, 0.02))
    r_far = _extract([good_frame_pose()] * 6, [far] * 6)
    assert r_far.frames_with_hand == 0


# --- F08c ------------------------------------------------------------------
def test_F08c_insufficient_hands():
    """<4 cuadros con mano asociada -> INSUFFICIENT_HANDS, quality_pass=False."""
    with_hand = good_frame_hands()
    without = make_hands_result()
    hands = [with_hand, with_hand, with_hand, without, without, without]
    r = _extract([good_frame_pose()] * 6, hands)
    assert r.frames_with_hand == 3
    assert "INSUFFICIENT_HANDS" in r.reason_codes
    assert r.quality_pass is False


# --- F08d (fallo deliberado) -----------------------------------------------
def test_F08d_no_shoulders_two_wrists_no_association():
    """Cuadro sin hombros válidos con 2 muñecas -> no asocia, no suma, no ambiguo.

    Fallo deliberado: si el cuadro suma a frames_with_hand o emite
    HAND_ASSOCIATION_AMBIGUOUS pese a no tener hombros, la prueba falla.
    """
    # 5 cuadros buenos (para pasar INSUFFICIENT_POSE) + 1 sin hombros pero con 2 manos.
    good_pose = good_frame_pose()
    # Hombros demasiado juntos (< 0.08*W) -> cuadro sin hombros válidos.
    bad_pose = make_pose_result(make_pose(left_shoulder=(0.50, 0.40), right_shoulder=(0.51, 0.40)))
    poses = [good_pose] * 5 + [bad_pose]
    two_hands = make_hands_result(LEFT_WRIST_XY, RIGHT_WRIST_XY)
    hands = [two_hands] * 6
    r = _extract(poses, hands)
    # Sólo los 5 cuadros con hombros válidos suman mano.
    assert r.valid_shoulders == 5
    assert r.frames_with_hand == 5, "el cuadro sin hombros no debe sumar mano"
    assert "HAND_ASSOCIATION_AMBIGUOUS" not in r.reason_codes


# --- F09a ------------------------------------------------------------------
def test_F09a_two_hands_clear_gap_not_ambiguous():
    """2 manos con gap>=0.05 -> asignación de menor costo, no ambigua."""
    # Manos exactamente sobre cada muñeca de pose -> una asignación domina.
    hands = make_hands_result(LEFT_WRIST_XY, RIGHT_WRIST_XY)
    r = _extract([good_frame_pose()] * 6, [hands] * 6)
    assert "HAND_ASSOCIATION_AMBIGUOUS" not in r.reason_codes
    assert r.frames_with_hand == 6
    assert r.quality_pass is True


# --- F09b ------------------------------------------------------------------
def test_F09b_two_hands_small_gap_ambiguous():
    """2 manos equidistantes (gap<0.05) -> HAND_ASSOCIATION_AMBIGUOUS, ambas ausentes."""
    # Ambas manos en el punto medio entre las dos muñecas de pose: ambas
    # asignaciones tienen el mismo costo -> gap=0 < 0.05.
    midpoint = ((LEFT_WRIST_XY[0] + RIGHT_WRIST_XY[0]) / 2,
                (LEFT_WRIST_XY[1] + RIGHT_WRIST_XY[1]) / 2)
    hands = make_hands_result(midpoint, midpoint)
    r = _extract([good_frame_pose()] * 6, [hands] * 6)
    assert "HAND_ASSOCIATION_AMBIGUOUS" in r.reason_codes
    assert r.frames_with_hand == 0  # ambas manos quedan ausentes


# --- F09c ------------------------------------------------------------------
def test_F09c_single_complete_assignment():
    """Sólo una asignación completa (el otro enlace >0.6) -> se usa sin gap.

    Mano A sobre la muñeca izquierda y mano B sobre la derecha: la asignación
    A->izq, B->der sobrevive; la permutación inversa (A->der, B->izq) cruza y
    cada enlace supera cost_max (las muñecas están ~1.2 de distancia en unidades
    de hombro), así que sólo UNA asignación es completa y se usa sin regla de gap.
    """
    hands = make_hands_result(LEFT_WRIST_XY, RIGHT_WRIST_XY)
    r = _extract([good_frame_pose()] * 6, [hands] * 6)
    assert "HAND_ASSOCIATION_AMBIGUOUS" not in r.reason_codes
    assert r.frames_with_hand == 6


# --- F10 (fallo deliberado) ------------------------------------------------
def test_F10_augment_preserves_shape_masks_zeros():
    """augment train_v1 preserva forma/máscaras/ceros; val/test sin augment.

    Fallo deliberado: si el augment tocara puntos con m=0 (los sacara de cero),
    la comprobación de ceros fallaría.
    """
    r = _extract([good_frame_pose()] * 6, [make_hands_result()] * 6)
    x = r.x  # manos ausentes -> muchos triples con m=0
    mask = x[:, 2::3]
    zero_positions = mask == 0.0

    aug = augment_train(x, seed=20261022, policy="train_v1")
    assert aug.shape == x.shape
    assert aug.dtype == np.float32
    # Máscaras intactas.
    np.testing.assert_array_equal(aug[:, 2::3], x[:, 2::3])
    # Donde m=0, x e y siguen EXACTAMENTE en cero.
    aug_x = aug[:, 0::3]
    aug_y = aug[:, 1::3]
    assert np.all(aug_x[zero_positions] == 0.0), "augment tocó un punto ausente (m=0)"
    assert np.all(aug_y[zero_positions] == 0.0), "augment tocó un punto ausente (m=0)"

    # val/test NO aplican augment: copia idéntica.
    val = augment_train(x, seed=20261022, policy="val")
    np.testing.assert_array_equal(val, x)


# --- F11 (fallo deliberado) ------------------------------------------------
@pytest.mark.parametrize(
    "n,expected",
    [
        (6, [0, 1, 2, 3, 4, 5]),
        (7, [0, 1, 2, 4, 5, 6]),
        (10, [0, 2, 4, 5, 7, 9]),
        (11, [0, 2, 4, 6, 8, 10]),
    ],
)
def test_F11_temporal_sampling_half_up(n, expected):
    """sample_indices half-up (§15). Fallo deliberado si usa round() nativo.

    round() nativo (half-to-even) daría, p. ej., N=11 k=1 -> round(2.0)=2 ok,
    pero empates .5 divergen; la tabla anclada detecta cualquier desviación.
    """
    assert sample_indices(n) == expected
    assert sample_indices(n)[0] == 0
    assert sample_indices(n)[-1] == n - 1


# --- F12 (fallo deliberado) ------------------------------------------------
def test_F12_temporal_order_matters():
    """Orden temporal corrupto cambia el resultado (fallo deliberado si invariante).

    Se construyen 6 cuadros con muñecas distintas por cuadro; barajar el orden
    temporal debe producir un array x distinto.
    """
    # Poses idénticas; manos en posiciones distintas por cuadro para que el orden
    # temporal se refleje en x.
    hand_centers = [
        (0.35, 0.50), (0.36, 0.52), (0.34, 0.55),
        (0.65, 0.50), (0.66, 0.52), (0.64, 0.55),
    ]
    poses = [good_frame_pose()] * 6
    hands_ordered = [make_hands_result(LEFT_WRIST_XY, c) for c in hand_centers]
    r_ordered = _extract(poses, hands_ordered)

    hands_shuffled = [hands_ordered[i] for i in [5, 0, 3, 1, 4, 2]]
    r_shuffled = _extract(poses, hands_shuffled)

    assert not np.array_equal(r_ordered.x, r_shuffled.x), (
        "el pipeline es invariante al orden temporal (defecto no detectado)"
    )
