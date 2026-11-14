import numpy as np

from src.lsc_models.sensitivity import drop_hands


def test_drop_hands_zeroes_only_hand_points():
    X = np.ones((2, 6, 147), dtype=np.float32)
    out = drop_hands(X, 0.5, "both", np.random.default_rng(1))
    assert np.array_equal(out[:, :, : 7 * 3], X[:, :, : 7 * 3])  # pose intacta
    assert (out == 0).any() and X.min() == 1  # no muta la entrada


def test_drop_zero_fraction_is_identity():
    X = np.random.default_rng(0).normal(size=(3, 6, 147)).astype(np.float32)
    assert np.array_equal(drop_hands(X, 0.0, "left", np.random.default_rng(1)), X)


def test_one_hand_only():
    X = np.ones((1, 6, 147), dtype=np.float32)
    out = drop_hands(X, 1.0, "left", np.random.default_rng(1))
    assert out[:, :, 28 * 3:].min() == 1 and out[:, :, 7 * 3: 28 * 3].max() == 0
