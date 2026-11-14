import numpy as np
import pytest

from src.lsc_models.architectures import build_model
from src.lsc_models.config import load_config

CFG = load_config()


@pytest.mark.parametrize("kind", ["bilstm", "transformer"])
def test_output_contract(kind):
    m = build_model(kind, CFG["models"][kind])
    assert m.input_shape == (None, 6, 147) and m.output_shape == (None, 10)
    p = m.predict(np.random.default_rng(0).normal(size=(4, 6, 147)).astype(np.float32), verbose=0)
    assert np.allclose(p.sum(1), 1.0, atol=1e-5) and p.min() >= 0


def test_parameter_budgets_comparable():
    a = build_model("bilstm", CFG["models"]["bilstm"]).count_params()
    b = build_model("transformer", CFG["models"]["transformer"]).count_params()
    assert 0.8 < a / b < 1.25, (a, b)


def test_unknown_architecture():
    with pytest.raises(ValueError):
        build_model("cnn", {})
