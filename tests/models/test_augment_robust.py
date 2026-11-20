"""Aumentación robusta v2 y perturbaciones compartidas (src/lsc_models/{perturb,augment_robust}.py)."""

from __future__ import annotations

import numpy as np
import pytest

from src.lsc_models import augment_robust as AR
from src.lsc_models import perturb as P


def make_base(both_hands: bool = True) -> np.ndarray:
    """Secuencia sintética: hombros, brazos y ambas manos en reposo abajo (como en el corpus)."""
    z = np.zeros((6, 49, 3), dtype=np.float32)
    z[:, 0] = [0.0, -0.8, 1.0]
    z[:, 1] = [0.5, 0.0, 1.0]     # hombro izq. (derecha de la imagen)
    z[:, 2] = [-0.5, 0.0, 1.0]
    z[:, 3] = [0.8, 0.8, 1.0]
    z[:, 4] = [-0.8, 0.8, 1.0]
    z[:, 5] = [0.7, 1.4, 1.0]
    z[:, 6] = [-0.7, 1.4, 1.0]
    rng = np.random.default_rng(1)
    for f in range(6):
        for i in range(21):
            # izquierda en reposo (abajo, y~1.4); derecha activa (cerca de la cara, y~0.3), como en el corpus
            z[f, 7 + i] = [0.7 + rng.normal(0, 0.1), 1.4 + rng.normal(0, 0.1) - 0.02 * i, 1.0]
            if both_hands:
                z[f, 28 + i] = [-0.4 + rng.normal(0, 0.1), 0.3 + rng.normal(0, 0.1) - 0.02 * i, 1.0]
    return z.reshape(6, 147)


# --- perturbaciones -------------------------------------------------------------------------------
def test_drop_hand_removes_the_whole_block():
    z = P.to_z(make_base())
    P.drop_hand(z, "right")
    assert not z[:, 28:49].any() and z[:, 7:28, 2].all()
    z = P.to_z(make_base())
    P.drop_hand(z, "left", frames=[0, 3])
    assert not z[[0, 3], 7:28].any() and z[[1, 2, 4, 5], 7:28, 2].all()


def test_crop_below_drops_hands_and_arm_points_beyond_the_line():
    z = P.to_z(make_base())
    P.crop_below(z, 1.0)  # muñeca izquierda (y~1.4) fuera; la mano derecha activa (y~0.3) se queda
    assert not z[:, 7:28].any() and z[:, 28:49, 2].all()
    assert not z[:, 5:7].any()          # muñecas de pose (y=1.4)
    assert z[:, 3:5, 2].all()           # codos (y=0.8) siguen
    assert z[:, 0:3, 2].all()           # nariz y hombros nunca se recortan
    z = P.to_z(make_base())
    P.crop_below(z, 2.0)                # nada supera 2.0
    np.testing.assert_array_equal(z, P.to_z(make_base()))


def test_crop_is_per_hand_using_the_wrist():
    z = P.to_z(make_base())
    z[:, 7, 1] = 0.3                     # muñeca izquierda arriba (se queda)
    z[:, 28, 1] = 1.4                    # muñeca derecha abajo (fuera); el resto de sus puntos no importa
    P.crop_below(z, 1.0)
    assert z[:, 7:28, 2].all() and not z[:, 28:49].any()   # se quita la mano completa, no punto a punto


def test_affine_moves_only_present_points():
    z = P.to_z(make_base(both_hands=False))
    out = P.affine(z.copy(), scale=2.0, shift_x=0.1, shift_y=-0.2, deg=0.0)
    present = out[..., 2] > 0
    np.testing.assert_allclose(out[present][:, 0], z[present][:, 0] * 2.0 + 0.1, atol=1e-5)
    assert not out[~present].any()


def test_is_processable_matches_the_app_rules():
    assert P.is_processable(make_base())
    z = P.to_z(make_base())
    P.drop_hand(z, "left"); P.drop_hand(z, "right")
    assert not P.is_processable(P.to_x(z))
    z = P.to_z(make_base()); P.drop_hand(z, "left"); P.drop_hand(z, "right", frames=[0, 1, 2])
    assert not P.is_processable(P.to_x(z))          # solo 3 cuadros con mano
    z = P.to_z(make_base()); P.drop_hand(z, "left"); P.drop_hand(z, "right", frames=[0, 1])
    assert P.is_processable(P.to_x(z))              # 4 cuadros con mano
    z = P.to_z(make_base()); z[:1, 1:3] = 0.0
    assert P.is_processable(P.to_x(z))              # 5 cuadros con ambos hombros: justo basta
    z[:2, 1:3] = 0.0
    assert not P.is_processable(P.to_x(z))          # solo 4: no pasa


# --- aumentación ----------------------------------------------------------------------------------
def test_shape_dtype_and_mask_invariants():
    x = make_base()
    for seed in range(200):
        out = AR.augment_robust(x, seed)
        assert out.shape == (6, 147) and out.dtype == np.float32 and np.isfinite(out).all()
        z = out.reshape(6, 49, 3)
        assert set(np.unique(z[..., 2])) <= {0.0, 1.0}
        assert not z[z[..., 2] == 0].any()          # ausentes = ceros exactos


def test_deterministic_by_seed_and_does_not_mutate_input():
    x = make_base()
    before = x.copy()
    a, b = AR.augment_robust(x, 123), AR.augment_robust(x, 123)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(x, before)
    assert not np.array_equal(AR.augment_robust(x, 1), AR.augment_robust(x, 2))


def test_output_is_always_processable():
    x = make_base()
    assert all(P.is_processable(AR.augment_robust(x, s)) for s in range(1500))


def test_it_actually_simulates_the_webcam_failure_modes():
    x = make_base()
    n = 1500
    outs = [AR.augment_robust(x, s) for s in range(n)]
    left_gone = np.mean([not P.hand_frames(o)[0].any() for o in outs])
    right_gone = np.mean([not P.hand_frames(o)[1].any() for o in outs])
    one_hand_frames = np.mean([(P.hand_frames(o)[0] ^ P.hand_frames(o)[1]).any() for o in outs])
    assert left_gone > 0.3             # la mano en reposo (abajo) queda fuera de cuadro con frecuencia
    assert 0.05 < right_gone < 0.5     # la mano activa también puede faltar, pero menos
    assert one_hand_frames > 0.4       # y los cuadros con una sola mano son comunes
    # el v1 jamás hacía esto
    from src.lsc_features.augment import augment_train

    v1 = [augment_train(x, s) for s in range(300)]
    assert all(P.hand_frames(o)[0].all() and P.hand_frames(o)[1].all() for o in v1)


def test_unknown_inputs_without_hands_fall_back_gracefully():
    z = P.to_z(make_base()); P.drop_hand(z, "left"); P.drop_hand(z, "right")
    out = AR.augment_robust(P.to_x(z), 5)       # ya no es procesable: no debe romper ni colgarse
    assert out.shape == (6, 147)


def test_strength_zero_leaves_geometry_untouched_except_noise():
    x = make_base()
    prm = {**AR.DEFAULT_PARAMS, "p_noise": 0.0}
    out = AR._once(x, np.random.default_rng(0), prm, 0.0)
    np.testing.assert_array_equal(out, x)
