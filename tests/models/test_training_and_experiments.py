import json

import numpy as np
import pytest

from src.lsc_models.ablation import (
    load_runs, select_configuration, summarize, write_freeze, write_tables,
)
from src.lsc_models.config import expand_runs, load_config
from src.lsc_models.data import Bundle
from src.lsc_models.experiments import is_complete, run_all
from src.lsc_models.train import train_one_run

from .conftest import FAST


@pytest.fixture(scope="module")
def all_runs(tmp_path_factory, synthetic_bundle_path):
    out = tmp_path_factory.mktemp("out")
    run_all("configs/train/experiments.yaml", out, bundle_path=synthetic_bundle_path, overrides=FAST)
    return out


def test_twelve_runs_written(all_runs):
    dirs = sorted(p.name for p in (all_runs / "runs").iterdir())
    assert len(dirs) == 12
    for d in dirs:
        for f in ("config.json", "metrics.json", "history.csv", "predictions_val.npz", "confusion_val.csv", "per_class_val.csv", "model.keras"):
            assert (all_runs / "runs" / d / f).is_file(), (d, f)


def test_runs_never_touch_test_split(all_runs):
    for p in (all_runs / "runs").glob("*/metrics.json"):
        m = json.loads(p.read_text())
        assert m["split_used"] == ["train", "val"]
        assert not any(k.startswith("n_test") for k in m["data"])
        assert "test" not in m


def test_synthetic_runs_flagged_not_evidence(all_runs):
    for p in (all_runs / "runs").glob("*/metrics.json"):
        m = json.loads(p.read_text())
        assert m["evidence_valid"] is False and m["data_source"] == "synthetic_contract_fixture"


def test_resume_skips_complete_runs(all_runs, synthetic_bundle_path, capsys):
    run_all("configs/train/experiments.yaml", all_runs, bundle_path=synthetic_bundle_path, overrides=FAST)
    assert capsys.readouterr().out.count("[skip]") == 12


def test_resume_reruns_when_config_changes(all_runs):
    cfg = load_config(overrides={**FAST, "training": {**FAST["training"], "max_epochs": 4}})
    assert not is_complete(expand_runs(cfg)[0], all_runs / "runs")


def test_tables_and_selection_do_not_claim_goals_on_synthetic(all_runs):
    cfg = load_config()
    sel = write_tables(load_runs(all_runs / "runs"), all_runs, cfg)
    assert sel["evidence_valid"] is False
    assert sel["goals_vs_validation"]["evaluated"] is False
    assert sel["deployed_run"].endswith("__seed17")
    for f in ("ablation_summary.csv", "per_seed.csv", "per_class_by_seed.csv", "selection.json"):
        assert (all_runs / f).is_file()


def test_freeze_refused_for_synthetic_runs(all_runs):
    with pytest.raises(RuntimeError, match="sintéticos"):
        write_freeze(all_runs, load_config())


def test_determinism_same_seed_same_predictions(tmp_path, bundle):
    cfg = load_config(overrides=FAST)
    spec = next(r for r in expand_runs(cfg) if r.run_id == "bilstm_aug__seed17")
    a = train_one_run(spec, bundle, tmp_path / "a")
    b = train_one_run(spec, bundle, tmp_path / "b")
    pa = np.load(tmp_path / "a" / spec.run_id / "predictions_val.npz")["probs"]
    pb = np.load(tmp_path / "b" / spec.run_id / "predictions_val.npz")["probs"]
    assert np.allclose(pa, pb, atol=1e-6)
    assert a["best_epoch"] == b["best_epoch"]


def _row(name, f1, p95):
    return {"config": name, "val_macro_f1_processable_mean": f1, "p95_latency_ms_mean": p95}


def test_selection_picks_best_f1_when_clear():
    s = select_configuration([_row("a", 0.80, 50), _row("b", 0.70, 5)])
    assert s["chosen_config"] == "a"


def test_selection_tie_picks_lowest_p95():
    s = select_configuration([_row("a", 0.805, 50), _row("b", 0.800, 5), _row("c", 0.70, 1)])
    assert s["chosen_config"] == "b"


def test_selection_tie_boundary_is_inclusive():
    s = select_configuration([_row("a", 0.80, 50), _row("b", 0.79, 5)])
    assert s["chosen_config"] == "b"
