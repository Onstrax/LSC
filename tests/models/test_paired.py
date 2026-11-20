import json

import numpy as np
import pytest

from src.lsc_models.paired import mcnemar_exact, paired_comparison, write_paired_comparison


def _npz(path, y, pred, qp=None, ids=None):
    n = len(y)
    probs = np.full((n, 10), 0.01, dtype=np.float32)
    probs[np.arange(n), pred] = 0.91
    np.savez(path, sample_id=ids if ids is not None else np.arange(n).astype(str), y_true=np.asarray(y),
             probs=probs, quality_pass=np.ones(n, bool) if qp is None else np.asarray(qp))


def test_mcnemar_exact_known_values():
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(5, 5) == 1.0
    assert mcnemar_exact(9, 1) == pytest.approx(0.021484375)     # 2 * (1 + 10) / 1024
    assert mcnemar_exact(1, 9) == mcnemar_exact(9, 1)


def test_paired_counts_and_difference(tmp_path):
    y = np.arange(20) % 10
    a_pred = y.copy(); a_pred[:4] = (y[:4] + 1) % 10           # A falla las 4 primeras
    b_pred = y.copy(); b_pred[2:6] = (y[2:6] + 1) % 10          # B falla las muestras 2..5
    _npz(tmp_path / "a.npz", y, a_pred); _npz(tmp_path / "b.npz", y, b_pred)
    r = paired_comparison(tmp_path / "a.npz", tmp_path / "b.npz", n_boot=200)
    assert r["n_processable"] == 20 and r["correct_a"] == 16 and r["correct_b"] == 16
    assert r["only_a_correct"] == 2 and r["only_b_correct"] == 2 and r["both_correct"] == 14 and r["none_correct"] == 2
    assert r["difference_b_minus_a"] == 0.0 and r["mcnemar_exact_p_two_sided"] == 1.0
    lo, hi = r["bootstrap_ci95"]
    assert lo <= 0.0 <= hi


def test_only_processable_samples_count(tmp_path):
    y = np.arange(10) % 10
    qp = np.array([True] * 8 + [False] * 2)
    _npz(tmp_path / "a.npz", y, (y + 1) % 10, qp); _npz(tmp_path / "b.npz", y, y, qp)
    r = paired_comparison(tmp_path / "a.npz", tmp_path / "b.npz", n_boot=100)
    assert r["n_processable"] == 8 and r["correct_a"] == 0 and r["correct_b"] == 8


def test_refuses_different_samples(tmp_path):
    y = np.arange(6)
    _npz(tmp_path / "a.npz", y, y); _npz(tmp_path / "b.npz", y, y, ids=np.array(list("uvwxyz")))
    with pytest.raises(ValueError, match="mismas muestras"):
        paired_comparison(tmp_path / "a.npz", tmp_path / "b.npz")


def test_write_creates_json_and_never_needs_a_model(tmp_path):
    y = np.arange(10) % 10
    for d in ("r1", "r2"):
        (tmp_path / d).mkdir()
        _npz(tmp_path / d / "predictions_test_deployed.npz", y, y)
    out = write_paired_comparison(tmp_path / "r1", tmp_path / "r2")
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["schema_version"] == "lsc.paired_test_comparison.v1" and "no se reevaluó" in doc["source"]
