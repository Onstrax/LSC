import numpy as np

from src.lsc_models import metrics as M
from src.lsc_models.baseline import MajorityClassifier


def test_macro_f1_perfect_and_zero_support_classes():
    y = np.array([0, 1, 2])
    assert M.macro_f1(y, y) == np.float64(3 / 10)  # 7 clases sin soporte cuentan 0


def test_integral_counts_rejections_as_errors():
    y = np.array([0, 1, 2, 3])
    probs = np.eye(10, dtype=np.float32)[y]
    qp = np.array([True, True, False, True])
    ev = M.evaluate_split(y, probs, qp)
    assert ev["accuracy_processable"] == 1.0
    assert ev["accuracy_integral"] == 0.75
    assert ev["quality_coverage"] == 0.75


def test_majority_baseline_tie_breaks_lowest_index():
    maj = MajorityClassifier().fit(np.array([3, 3, 1, 1]))
    assert maj.majority_class == 1
    p = maj.predict_proba(4)
    assert p.shape == (4, 10) and np.all(p.sum(1) == 1)


def test_majority_baseline_is_weak():
    y = np.repeat(np.arange(10), 5)
    maj = MajorityClassifier().fit(y)
    ev = M.evaluate_split(y, maj.predict_proba(len(y)), np.ones(len(y), bool))
    assert ev["accuracy_processable"] == 0.1 and ev["macro_f1_processable"] < 0.05
