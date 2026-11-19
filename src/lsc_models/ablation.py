"""Análisis de ablación, selección y congelación (A3).

Todo se calcula desde los archivos guardados por cada ejecución
(``metrics.json``, ``predictions_val.npz``, ``history.csv``): las tablas y
gráficas del artículo son reproducibles sin reentrenar.

Regla de selección (plan §6, fijada antes del test):
  1. media de F1 macro de validación (procesables) sobre las 3 semillas;
  2. si la diferencia con la mejor es <= 0,01, gana el menor p95 de inferencia
     (media de p95 sobre semillas, mismo equipo);
  3. el artefacto desplegado es la semilla 17 de la configuración elegida.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from . import metrics as M
from .config import REQUIRED_CONFIGS, REQUIRED_SEEDS

CONF_ORDER = list(REQUIRED_CONFIGS)


def load_runs(runs_dir: str | Path) -> list[dict]:
    runs_dir = Path(runs_dir)
    out = []
    for name in CONF_ORDER:
        for seed in REQUIRED_SEEDS:
            p = runs_dir / f"{name}__seed{seed}" / "metrics.json"
            if not p.is_file():
                raise FileNotFoundError(f"falta la ejecución {name}__seed{seed} ({p})")
            out.append(json.loads(p.read_text(encoding="utf-8")))
    return out


def _mean_std(vals) -> tuple[float, float]:
    a = np.asarray(vals, dtype=np.float64)
    return float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else 0.0


def summarize(runs: list[dict]) -> list[dict]:
    rows = []
    for name in CONF_ORDER:
        rs = [r for r in runs if r["name"] == name]
        row = {"config": name, "model": rs[0]["model"], "augment": rs[0]["augment"], "n_seeds": len(rs), "n_params": rs[0]["n_params"]}
        for key, getter in {
            "val_accuracy_processable": lambda r: r["val"]["accuracy_processable"],
            "val_macro_f1_processable": lambda r: r["val"]["macro_f1_processable"],
            "val_accuracy_integral": lambda r: r["val"]["accuracy_integral"],
            "val_macro_f1_integral": lambda r: r["val"]["macro_f1_integral"],
            "val_loss": lambda r: r["val"]["loss_processable"],
            "train_accuracy": lambda r: r["train"]["accuracy"],
            "best_epoch": lambda r: r["best_epoch"],
            "p95_latency_ms": lambda r: r["latency"]["p95_ms"],
        }.items():
            m, s = _mean_std([getter(r) for r in rs])
            row[f"{key}_mean"], row[f"{key}_std"] = m, s
        rows.append(row)
    return rows


def select_configuration(summary: list[dict], tie_tolerance: float = 0.01) -> dict:
    best_f1 = max(r["val_macro_f1_processable_mean"] for r in summary)
    tied = [r for r in summary if best_f1 - r["val_macro_f1_processable_mean"] <= tie_tolerance + 1e-12]
    if len(tied) == 1:
        chosen, reason = tied[0], "mayor F1 macro medio de validación"
    else:
        chosen = min(tied, key=lambda r: (r["p95_latency_ms_mean"], r["config"]))
        reason = f"empate técnico (<= {tie_tolerance}) entre {[t['config'] for t in tied]}; gana menor p95 de inferencia"
    return {
        "chosen_config": chosen["config"],
        "reason": reason,
        "best_val_macro_f1_mean": best_f1,
        "candidates_within_tolerance": [t["config"] for t in tied],
        "tie_tolerance": tie_tolerance,
    }


def goals_check(summary_row: dict, goals: dict) -> dict:
    """Compara con las METAS (no son resultados). Se informa aunque no se cumplan."""
    acc, f1 = summary_row["val_accuracy_processable_mean"], summary_row["val_macro_f1_processable_mean"]
    return {
        "scope": "validación, muestras procesables, media de 3 semillas (NO es test)",
        "accuracy": {"value": acc, "goal": goals["accuracy"], "met": bool(acc >= goals["accuracy"])},
        "macro_f1": {"value": f1, "goal": goals["macro_f1"], "met": bool(f1 >= goals["macro_f1"])},
    }


def write_tables(runs: list[dict], out_dir: str | Path, cfg: dict) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = summarize(runs)
    _write_csv(out / "ablation_summary.csv", summary)

    per_seed = [
        {
            "config": r["name"], "seed": r["seed"], "best_epoch": r["best_epoch"], "n_params": r["n_params"],
            "val_accuracy_processable": r["val"]["accuracy_processable"],
            "val_macro_f1_processable": r["val"]["macro_f1_processable"],
            "val_accuracy_integral": r["val"]["accuracy_integral"],
            "val_macro_f1_integral": r["val"]["macro_f1_integral"],
            "quality_coverage_val": r["val"]["quality_coverage"],
            "baseline_majority_acc": r["baseline_majority_val"]["accuracy_processable"],
            "baseline_majority_macro_f1": r["baseline_majority_val"]["macro_f1_processable"],
            "p95_latency_ms": r["latency"]["p95_ms"],
        }
        for r in runs
    ]
    _write_csv(out / "per_seed.csv", per_seed)

    per_class = []
    for r in runs:
        with (out / "runs" / r["run_id"] / "per_class_val.csv").open(encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                per_class.append({"config": r["name"], "seed": r["seed"], **row})
    _write_csv(out / "per_class_by_seed.csv", per_class)

    sel = select_configuration(summary, float(cfg["selection"]["tie_tolerance"]))
    chosen_row = next(s for s in summary if s["config"] == sel["chosen_config"])
    sel["deployed_run"] = f"{sel['chosen_config']}__seed{cfg['deployed_seed']}"
    sel["evidence_valid"] = all(r["evidence_valid"] for r in runs)
    sel["goals_vs_validation"] = (
        goals_check(chosen_row, cfg["goals"])
        if sel["evidence_valid"]
        else {"evaluated": False, "reason": "datos sintéticos (synthetic_contract_fixture): no es evidencia de exactitud"}
    )
    sel["data_sha256"] = sorted({r["data"]["sha256"] for r in runs})
    (out / "selection.json").write_text(json.dumps(sel, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return sel


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


# ----------------------------------------------------------------------
# Gráficas (desde predicciones / historiales guardados)
# ----------------------------------------------------------------------
def make_plots(out_dir: str | Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(out_dir)
    runs_dir = out / "runs"
    figs = out / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    written = []

    # Curvas de aprendizaje: una figura por configuración, 3 semillas.
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=False)
    for ax, name in zip(axes.ravel(), CONF_ORDER):
        for seed in REQUIRED_SEEDS:
            h = np.genfromtxt(runs_dir / f"{name}__seed{seed}" / "history.csv", delimiter=",", names=True)
            ax.plot(h["epoch"], h["val_macro_f1"], label=f"val F1 s{seed}")
            ax.plot(h["epoch"], h["train_loss"], "--", alpha=0.5, label=f"train loss s{seed}")
        ax.set_title(name)
        ax.set_xlabel("época")
        ax.legend(fontsize=6)
    fig.tight_layout()
    p = figs / "learning_curves.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    written.append(p)

    # Matrices de confusión de validación (semilla 17) desde predicciones guardadas.
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    for ax, name in zip(axes.ravel(), CONF_ORDER):
        d = np.load(runs_dir / f"{name}__seed17" / "predictions_val.npz", allow_pickle=False)
        qp = d["quality_pass"]
        conf = M.confusion(d["y_true"][qp], np.argmax(d["probs"], 1)[qp])
        ax.imshow(conf, cmap="Blues")
        ax.set_title(f"{name} (semilla 17, val)")
        ax.set_xlabel("predicha")
        ax.set_ylabel("real")
        for i in range(conf.shape[0]):
            for j in range(conf.shape[1]):
                if conf[i, j]:
                    ax.text(j, i, int(conf[i, j]), ha="center", va="center", fontsize=7)
    fig.tight_layout()
    p = figs / "confusion_val_seed17.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    written.append(p)

    # F1 macro por configuración y semilla.
    fig, ax = plt.subplots(figsize=(7, 4))
    for k, name in enumerate(CONF_ORDER):
        vals = [json.loads((runs_dir / f"{name}__seed{s}" / "metrics.json").read_text())["val"]["macro_f1_processable"] for s in REQUIRED_SEEDS]
        ax.scatter([k] * 3, vals, label=None)
        ax.hlines(np.mean(vals), k - 0.25, k + 0.25)
    ax.set_xticks(range(4))
    ax.set_xticklabels(CONF_ORDER, rotation=20, fontsize=8)
    ax.set_ylabel("F1 macro (validación, procesables)")
    fig.tight_layout()
    p = figs / "macro_f1_by_config_seed.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    written.append(p)
    return written


# ----------------------------------------------------------------------
# Congelación (18 nov.)
# ----------------------------------------------------------------------
def write_freeze(out_dir: str | Path, cfg: dict, freeze_path: str | Path | None = None) -> Path:
    """Congela decisiones: habilita el uso único del test (A4)."""
    out = Path(out_dir)
    sel_p = out / "selection.json"
    if not sel_p.is_file():
        raise FileNotFoundError("falta selection.json: ejecute `select` antes de congelar")
    sel = json.loads(sel_p.read_text(encoding="utf-8"))
    if not sel["evidence_valid"]:
        raise RuntimeError("las ejecuciones usan datos sintéticos: no se puede congelar con ellas")
    runs = load_runs(out / "runs")  # exige las 12
    hashes = {}
    for r in runs:
        mp = out / "runs" / r["run_id"] / "model.keras"
        hashes[r["run_id"]] = hashlib.sha256(mp.read_bytes()).hexdigest()
    doc = {
        "schema_version": "lsc.freeze.v1",
        "chosen_config": sel["chosen_config"],
        "deployed_run": sel["deployed_run"],
        "selection_rule": "val_macro_f1 medio; empate <=0.01 -> menor p95; semilla 17",
        "config_hashes": {r["run_id"]: r["config_hash"] for r in runs},
        "model_sha256": hashes,
        "data_sha256": sel["data_sha256"],
        "test_used": False,
    }
    path = Path(freeze_path) if freeze_path else out / "freeze.json"
    path.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path
