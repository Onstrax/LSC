"""Pipeline del modelo v2: configuración, entrenamiento de punta a punta (sintético), selección y sellado del test."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.lsc_models import perturb as P
from src.lsc_models import robustness as R
from src.lsc_models.ablation import load_runs, select_configuration, write_freeze, write_tables
from src.lsc_models.config import ConfigError, config_names, expand_runs, load_config, validate_config
from src.lsc_models.experiments import run_all
from src.lsc_models.train import augment_epoch

from .conftest import FAST

V2 = "configs/train/experiments_v2.yaml"
V1 = "configs/train/experiments.yaml"
FREEZE_V1 = Path("reports/models/freeze.json")


# --- v1 NO cambió ---------------------------------------------------------------------------------
@pytest.mark.skipif(not FREEZE_V1.is_file(), reason="no hay freeze.json de v1 en este entorno")
def test_v1_config_hashes_still_match_the_frozen_ones():
    """Si esto falla, `final-eval` de v1 quedaría invalidado: los config_hash deben ser idénticos a los congelados."""
    frozen = json.loads(FREEZE_V1.read_text(encoding="utf-8"))["config_hashes"]
    now = {r.run_id: r.config_hash() for r in expand_runs(load_config(V1))}
    assert now == frozen


def test_v1_runs_carry_no_evaluation_block():
    assert all(not r.evaluation for r in expand_runs(load_config(V1)))
    assert all("evaluation" not in r.to_dict() for r in expand_runs(load_config(V1)))


def test_v1_policy_dispatch_is_unchanged():
    from src.lsc_features.augment import augment_train

    X = np.random.default_rng(0).normal(0, 1, (4, 6, 147)).astype(np.float32)
    got = augment_epoch(X, 17, 3, "train_v1")
    from src.lsc_models.train import _aug_seed

    want = np.stack([augment_train(X[i], _aug_seed(17, 3, i), "train_v1") for i in range(4)])
    np.testing.assert_array_equal(got, want)


# --- config v2 ------------------------------------------------------------------------------------
def test_v2_config_loads_and_expands_six_runs():
    cfg = load_config(V2)
    assert config_names(cfg) == ["bilstm_robust", "transformer_robust"]
    runs = expand_runs(cfg)
    assert [r.run_id for r in runs] == [f"{n}__seed{s}" for n in ("bilstm_robust", "transformer_robust") for s in (17, 29, 43)]
    assert all(r.evaluation["test_sealed"] and r.evaluation["robust_val"] for r in runs)
    assert "evaluation" in runs[0].to_dict()                      # v2 sí lo incluye en el hash
    assert runs[0].config_hash() != expand_runs(load_config(V1))[0].config_hash()


def test_v2_uses_the_same_model_sizes_and_budget_as_v1():
    a, b = load_config(V1), load_config(V2)
    assert a["models"] == b["models"]
    for k in ("optimizer", "learning_rate", "batch_size", "max_epochs", "patience", "min_epochs", "restore_best"):
        assert a["training"][k] == b["training"][k]


def test_validate_rejects_unknown_policy_and_wrong_config_set():
    cfg = load_config(V2)
    bad = json.loads(json.dumps(cfg))
    bad["augmentation"]["policy"] = "train_v9"
    with pytest.raises(ConfigError, match="política"):
        validate_config(bad)
    bad = json.loads(json.dumps(cfg))
    bad["required_configs"] = ["otra"]
    with pytest.raises(ConfigError, match="configuraciones"):
        validate_config(bad)
    v1 = load_config(V1)
    v1.pop("required_configs", None)
    v1["configs"].pop("bilstm_aug")
    with pytest.raises(ConfigError):
        validate_config(v1)


def test_v2_dispatch_uses_the_robust_policy():
    X = np.stack([np.random.default_rng(i).normal(0, 1, 882) for i in range(3)]).reshape(3, 6, 147).astype(np.float32)
    X.reshape(3, 6, 49, 3)[..., 2] = 1.0
    out = augment_epoch(X, 17, 1, "train_v2_robust", {"p_drop_hand": 1.0})
    assert out.shape == X.shape and not np.array_equal(out, X)


# --- selección y sellado --------------------------------------------------------------------------
def _row(name, mixed, p95, f1=0.9):
    return {"config": name, "val_macro_f1_processable_mean": f1, "val_mixed_f1_mean": mixed, "p95_latency_ms_mean": p95}


def test_selection_uses_the_declared_metric():
    rows = [_row("a", 0.70, 50, f1=0.95), _row("b", 0.85, 60, f1=0.80)]
    assert select_configuration(rows, 0.01)["chosen_config"] == "a"                      # v1: F1 original
    sel = select_configuration(rows, 0.01, "val_mixed_f1")
    assert sel["chosen_config"] == "b" and sel["selection_metric"] == "val_mixed_f1"
    assert "selection_metric" not in select_configuration(rows, 0.01)                    # v1: selection.json sin claves nuevas


def test_selection_tie_on_mixed_metric_picks_lowest_p95():
    rows = [_row("a", 0.850, 60), _row("b", 0.845, 40)]
    assert select_configuration(rows, 0.01, "val_mixed_f1")["chosen_config"] == "b"


# --- robustez -------------------------------------------------------------------------------------
def _val_like(n=20):
    rng = np.random.default_rng(3)
    z = np.zeros((n, 6, 49, 3), dtype=np.float32)
    z[..., 0:3, :] = [0.0, 0.0, 1.0]
    z[:, :, 1] = [0.5, 0.0, 1.0]
    z[:, :, 2] = [-0.5, 0.0, 1.0]
    z[:, :, 5] = [0.7, 1.4, 1.0]
    z[:, :, 6] = [-0.7, 0.4, 1.0]
    z[:, :, 7:28] = np.concatenate([rng.normal(0.7, 0.1, (n, 6, 21, 1)), rng.normal(1.4, 0.1, (n, 6, 21, 1)), np.ones((n, 6, 21, 1))], -1)
    z[:, :, 28:49] = np.concatenate([rng.normal(-0.5, 0.1, (n, 6, 21, 1)), rng.normal(0.3, 0.1, (n, 6, 21, 1)), np.ones((n, 6, 21, 1))], -1)
    return z.reshape(n, 6, 147), rng.integers(0, 10, n)


def test_scenarios_are_deterministic_and_filtered_by_app_quality():
    X, _ = _val_like()
    a, b = R.build_scenarios(X, 7), R.build_scenarios(X, 7)
    assert list(a) == list(R.SCENARIO_NAMES)
    for k in a:
        np.testing.assert_array_equal(a[k][0], b[k][0])
        np.testing.assert_array_equal(a[k][1], b[k][1])
    Xp, ok = a["sin_mano_der"]
    assert not P.hand_frames(Xp[0])[1].any() and P.hand_frames(Xp[0])[0].all()   # quitó la derecha y dejó la izquierda
    assert ok.all()
    # sin ambas manos no es aceptable y se excluye (no cuenta como acierto ni fallo)
    z = P.to_z(X[0]); P.drop_hand(z, "left"); P.drop_hand(z, "right")
    assert not P.is_processable(P.to_x(z))


def test_evaluate_scenarios_with_a_fake_model_and_mixed_f1():
    X, y = _val_like()
    scen = R.build_scenarios(X, 7)

    def oracle(a):  # "modelo" perfecto: devuelve la etiqueta por posición; solo comprueba el cableado
        out = np.full((len(a), 10), 0.01, dtype=np.float32)
        out[:, 3] = 0.91
        return out / out.sum(1, keepdims=True)

    res = R.evaluate_scenarios(oracle, np.full(len(y), 3), scen)
    assert res["mean"]["accuracy"] == 1.0 and res["mean"]["conf_median"] > 0.9
    assert all(v["n"] > 0 for v in res["scenarios"].values())
    assert R.mixed_f1(0.8, {"mean": {"macro_f1": 0.6}}) == pytest.approx(0.7)
    assert R.mixed_f1(0.8, {"mean": {"macro_f1": float("nan")}}) == 0.8             # nunca compara contra NaN


# --- punta a punta con datos sintéticos (solo tubería; NO es evidencia) -----------------------------
@pytest.fixture(scope="module")
def v2_runs(tmp_path_factory, synthetic_bundle_path):
    out = tmp_path_factory.mktemp("v2out")
    run_all(V2, out, bundle_path=synthetic_bundle_path, overrides=FAST)
    return out


def test_v2_writes_six_runs_with_robust_metrics(v2_runs):
    dirs = sorted(p.name for p in (v2_runs / "runs").iterdir())
    assert len(dirs) == 6 and all(d.split("__")[0] in ("bilstm_robust", "transformer_robust") for d in dirs)
    for d in dirs:
        m = json.loads((v2_runs / "runs" / d / "metrics.json").read_text(encoding="utf-8"))
        assert m["split_used"] == ["train", "val"] and "test" not in m            # nunca toca el test
        assert m["evidence_valid"] is False                                         # sintético
        assert set(m["val_robust"]["scenarios"]) == set(R.SCENARIO_NAMES)
        assert 0.0 <= m["val_mixed_f1"] <= 1.0
        hist = (v2_runs / "runs" / d / "history.csv").read_text(encoding="utf-8").splitlines()[0]
        assert "val_robust_f1" in hist and "val_mixed_f1" in hist


def test_v2_tables_selection_and_plots_use_the_v2_names(v2_runs):
    cfg = load_config(V2)
    names = config_names(cfg)
    sel = write_tables(load_runs(v2_runs / "runs", names), v2_runs, cfg)
    assert sel["chosen_config"] in names and sel["selection_metric"] == "val_mixed_f1"
    assert sel["deployed_run"] == f"{sel['chosen_config']}__seed17"
    header = (v2_runs / "ablation_summary.csv").read_text(encoding="utf-8").splitlines()[0]
    assert "val_mixed_f1_mean" in header and "val_robust_macro_f1_mean" in header
    from src.lsc_models.ablation import make_plots

    assert len(make_plots(v2_runs, names)) == 3


def test_v2_export_fits_the_threshold_on_original_plus_perturbed_validation(v2_runs, synthetic_bundle_path, tmp_path):
    from src.lsc_models.export import export_artifacts

    cfg = load_config(V2)
    write_tables(load_runs(v2_runs / "runs", config_names(cfg)), v2_runs, cfg)
    out = export_artifacts(v2_runs, tmp_path / "pkg", synthetic_bundle_path, allow_synthetic=True, models_dir=tmp_path / "sin_task")
    pol = json.loads((out / "thresholds.json").read_text(encoding="utf-8"))
    assert "perturbada" in pol["fitted_on"] and pol["n_perturbed_samples"] > 0 and "fit_original_only" in pol
    assert pol["provisional"] is True and pol["requires_confirmation"] is True and pol["negatives_rejection_measured"] is False
    assert "sintética" in (out / "model_card.md").read_text(encoding="utf-8").lower()      # sintético manda sobre v2
    assert json.loads((out / "manifest.json").read_text(encoding="utf-8"))["test_used"] is False


def test_model_card_of_a_real_v2_says_the_test_is_sealed():
    from src.lsc_models.export import _model_card

    m = {"model": "bilstm", "augment": True, "seed": 17, "n_params": 1, "val_robust": {"mean": {}},
         "val": {"accuracy_processable": 0.9, "macro_f1_processable": 0.9, "n_processable": 63}}
    fit = {"threshold": 0.9, "coverage": 0.8, "accuracy_accepted": 0.9, "goals_met": True}
    card = _model_card(m, fit, "v", False, "bilstm_robust__seed17")
    assert "v2" in card and "SELLADO" in card
    m.pop("val_robust")
    assert "SELLADO" not in _model_card(m, fit, "v", False, "x")                          # v1 no cambia


def test_v2_can_never_be_frozen_so_its_test_stays_sealed(v2_runs):
    with pytest.raises(RuntimeError, match="SELLADO"):
        write_freeze(v2_runs, load_config(V2))
    assert not (v2_runs / "freeze.json").exists()
