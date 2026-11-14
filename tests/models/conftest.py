import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import pytest

from src.lsc_models.data import Bundle, make_synthetic_contract_fixture

FAST = {
    "training": {"max_epochs": 3, "patience": 2, "min_epochs": 1},
    "latency": {"warmup": 1, "repeats": 5},
}


@pytest.fixture(scope="session")
def synthetic_bundle_path(tmp_path_factory):
    # synthetic_contract_fixture: SOLO interfaces, nunca evidencia de exactitud
    return make_synthetic_contract_fixture(tmp_path_factory.mktemp("syn") / "syn.npz", n_signers=14)


@pytest.fixture()
def bundle(synthetic_bundle_path, tmp_path):
    return Bundle(synthetic_bundle_path, freeze_file=tmp_path / "freeze.json")
