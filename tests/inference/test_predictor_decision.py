import numpy as np
import pytest

from src.lsc_inference import DecisionPolicy, Predictor, PredictorInputError, SimulatedPredictor
from src.lsc_models.architectures import build_model
from src.lsc_models.config import load_config

CFG = load_config()
X = np.random.default_rng(3).normal(size=(5, 6, 147)).astype(np.float32)


@pytest.fixture(scope="module", params=["bilstm", "transformer"])
def saved(request, tmp_path_factory):
    m = build_model(request.param, CFG["models"][request.param])
    p = tmp_path_factory.mktemp("m") / "model.keras"
    m.save(p)
    return m, p


def test_exported_prediction_matches_in_memory(saved):
    model, path = saved
    ref = model.predict(X, verbose=0)
    got = Predictor(path).predict(X)
    assert got.shape == (5, 10) and got.dtype == np.float32
    assert np.allclose(got, ref, atol=1e-5)  # tolerancia acordada: 1e-5


def test_predictor_validates_input(saved):
    pred = Predictor(saved[1])
    for bad in (np.zeros((2, 5, 147)), np.zeros((2, 6, 146)), np.full((1, 6, 147), np.nan), np.zeros((0, 6, 147))):
        with pytest.raises(PredictorInputError):
            pred.predict(bad)


def test_predictor_accepts_single_sequence(saved):
    assert Predictor(saved[1]).predict(X[0]).shape == (1, 10)


def test_missing_model_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        Predictor(tmp_path / "x.keras")


def test_simulated_is_deterministic_and_flagged():
    sim = SimulatedPredictor()
    a, b = sim.predict(X), sim.predict(X)
    assert np.array_equal(a, b) and np.allclose(a.sum(1), 1, atol=1e-5)
    assert sim.is_simulated is True


def test_policy_accept_and_reject():
    pol = DecisionPolicy({"policy_version": "t1", "threshold": 0.6})
    hi = np.array([0.7] + [0.1] * 3 + [0.0] * 6, dtype=np.float32)
    lo = np.full(10, 0.1, dtype=np.float32)
    assert pol.decide(hi).accepted and pol.decide(hi).label_id == 0
    d = pol.decide(lo)
    assert not d.accepted and d.label_id is None and d.reason_codes == ["LOW_CONFIDENCE"]


def test_policy_threshold_is_inclusive_and_tie_lowest_index():
    pol = DecisionPolicy({"policy_version": "t1", "threshold": 0.5})
    p = np.array([0.5, 0.5] + [0.0] * 8, dtype=np.float32)
    d = pol.decide(p)
    assert d.accepted and d.label_id == 0


def test_policy_requires_version_and_valid_mode():
    with pytest.raises(ValueError):
        DecisionPolicy({"threshold": 0.5})
    with pytest.raises(ValueError):
        DecisionPolicy({"policy_version": "x", "mode": "other"})


def test_api_real_adapters_accept_our_classes(saved, tmp_path):
    """Compatibilidad con api/adapters/real.py: Predictor(path), DecisionPolicy(dict)."""
    from api.adapters.real import verify_probabilities

    verify_probabilities(Predictor(saved[1]).predict(X), batch=5)


def test_api_realpredictor_loads_our_artifact(saved, tmp_path):
    """RealPredictor/RealDecisionAdapter de Juan funcionan con nuestras clases."""
    import json
    import shutil

    from api.adapters.real import RealDecisionAdapter, RealPredictor

    shutil.copy(saved[1], tmp_path / "model.keras")
    (tmp_path / "manifest.json").write_text(json.dumps({"model_version": "t", "feature_spec_hash": "h"}))
    (tmp_path / "thresholds.json").write_text(json.dumps({"policy_version": "t1", "threshold": 0.0}))
    probs = RealPredictor(tmp_path).predict(X[:1])
    assert probs.shape == (1, 10)
    assert RealDecisionAdapter(tmp_path).decide(probs).accepted
