import copy

import pytest

from src.lsc_models.config import ConfigError, expand_runs, load_config, validate_config

CFG = load_config()


def test_twelve_runs_four_configs_three_seeds():
    runs = expand_runs(CFG)
    assert len(runs) == 12
    assert {r.seed for r in runs} == {17, 29, 43}
    assert {r.name for r in runs} == {"bilstm_noaug", "bilstm_aug", "transformer_noaug", "transformer_aug"}
    assert len({r.run_id for r in runs}) == 12


def test_same_budget_for_all_runs():
    runs = expand_runs(CFG)
    assert len({str(r.training) for r in runs}) == 1
    assert len({str(r.augmentation) for r in runs}) == 1


def test_config_hash_changes_with_budget():
    a = expand_runs(CFG)[0].config_hash()
    cfg2 = copy.deepcopy(CFG)
    cfg2["training"]["max_epochs"] += 1
    assert expand_runs(cfg2)[0].config_hash() != a


@pytest.mark.parametrize("mut", [
    lambda c: c.__setitem__("seeds", [1, 2, 3]),
    lambda c: c.__setitem__("deployed_seed", 29),
    lambda c: c["configs"].pop("bilstm_aug"),
])
def test_invalid_configs_rejected(mut):
    cfg = copy.deepcopy(CFG)
    mut(cfg)
    with pytest.raises(ConfigError):
        validate_config(cfg)
