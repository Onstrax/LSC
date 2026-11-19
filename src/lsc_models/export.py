"""Paquete de artefactos del modelo para la aplicación (``artifacts/``).

Genera lo que la API en modo real exige: model.keras, labels.json, feature_spec.json,
preprocessing.json, thresholds.json, model_card.md y manifest.json (con hashes y versiones).
Copia además los .task de Ortiz (artifacts/models/ -> artifacts/) si existen.

Reglas:
- La ejecución exportada es la desplegable de ``selection.json`` (semilla 17 de la
  configuración elegida) salvo ``--run`` explícito.
- Con resultados sintéticos se niega, salvo ``allow_synthetic=True``: entonces el
  paquete queda marcado ``SYNTHETIC-DEMO`` y NO es un modelo de LSC.
- Verifica paridad: lo que predice el modelo cargado desde el paquete coincide con las
  predicciones de validación guardadas (tolerancia 1e-5).
- El umbral se ajusta SOLO con validación (cobertura >= 80 %; el más estricto que la cumpla) y se
  informa si la exactitud entre aceptados llega a 90 %. Es PROVISIONAL: A4 lo recalibra con
  webcam de desarrollo y mide negativos aparte. Mientras tanto la interfaz debe tratar la salida
  como sugerencia que requiere confirmación.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

from .data import Bundle

CONTRACTS = Path(__file__).resolve().parents[2] / "contracts"
TASKS = ("pose_landmarker.task", "hand_landmarker.task")
COVERAGE_GOAL, ACCURACY_GOAL = 0.80, 0.90


class ExportError(RuntimeError):
    pass


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fit_threshold(y_true: np.ndarray, probs: np.ndarray, quality_pass: np.ndarray) -> dict:
    """Umbral de máximo-softmax ajustado con validación (procesables)."""
    qp = np.asarray(quality_pass, dtype=bool)
    y, p = np.asarray(y_true)[qp], np.asarray(probs)[qp]
    if len(y) == 0:
        raise ExportError("la validación no tiene muestras procesables: no se puede ajustar el umbral")
    conf, pred = p.max(axis=1), p.argmax(axis=1)
    best = {"threshold": 0.0, "coverage": 1.0, "accuracy_accepted": float((pred == y).mean())}
    for t in sorted(set(conf.tolist())):
        acc_mask = conf >= t
        cov = float(acc_mask.mean())
        if cov >= COVERAGE_GOAL and t >= best["threshold"]:
            best = {"threshold": float(t), "coverage": cov, "accuracy_accepted": float((pred[acc_mask] == y[acc_mask]).mean())}
    best.update(
        n_val_processable=int(len(y)),
        coverage_goal=COVERAGE_GOAL,
        accuracy_goal=ACCURACY_GOAL,
        goals_met=bool(best["coverage"] >= COVERAGE_GOAL and best["accuracy_accepted"] >= ACCURACY_GOAL),
    )
    return best


def export_artifacts(
    reports_dir: str | Path,
    out_dir: str | Path,
    bundle_path: str | Path,
    *,
    run_id: str | None = None,
    allow_synthetic: bool = False,
    force: bool = False,
    models_dir: str | Path = "artifacts/models",
) -> Path:
    reports, out = Path(reports_dir), Path(out_dir)
    sel_p = reports / "selection.json"
    if run_id is None:
        if not sel_p.is_file():
            raise ExportError("falta selection.json (ejecute `aggregate`) o indique --run")
        run_id = json.loads(sel_p.read_text(encoding="utf-8"))["deployed_run"]
    run = reports / "runs" / run_id
    metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    synthetic = not metrics["evidence_valid"]
    if synthetic and not allow_synthetic:
        raise ExportError("la ejecución usa datos sintéticos: no se exporta como modelo real (use --allow-synthetic solo para una DEMO de integración)")
    if (out / "model.keras").exists() and not force:
        raise ExportError(f"{out}/model.keras ya existe; use --force para sobrescribir")

    bundle = Bundle(bundle_path)
    if bundle.sha256 != metrics["data"]["sha256"]:
        raise ExportError("el bundle indicado no es el mismo con el que se entrenó la ejecución (SHA-256 distinto)")

    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(run / "model.keras", out / "model.keras")

    # --- paridad: paquete exportado == predicciones guardadas ------------------
    from src.lsc_inference import Predictor

    va = bundle.load_split("val")
    saved = np.load(run / "predictions_val.npz", allow_pickle=False)
    got = Predictor(out / "model.keras").predict(va.X)
    max_diff = float(np.abs(got - saved["probs"]).max())
    if max_diff > 1e-5:
        raise ExportError(f"el modelo exportado no reproduce las predicciones guardadas (máx |Δ|={max_diff:.2e} > 1e-5)")

    # --- artefactos de contrato -------------------------------------------------
    spec = json.loads((CONTRACTS / "feature_spec.json").read_text(encoding="utf-8"))
    shutil.copy2(CONTRACTS / "feature_spec.json", out / "feature_spec.json")
    entries = json.loads((CONTRACTS / "labels.json").read_text(encoding="utf-8"))["labels"]
    # La API de Juan espera TEXTOS en "labels"; se conservan los objetos del contrato en "entries".
    labels_doc = {"schema_version": "lsc.labels.v1", "labels": [e["display_text"] for e in entries], "entries": entries}
    (out / "labels.json").write_text(json.dumps(labels_doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    prep = {
        "schema_version": "lsc.preprocessing.v1",
        "type": "identity",
        "note": "Sin transformación adicional: el extractor (Ortiz) ya normaliza por hombros.",
        "input_shape": [6, 147],
        "dtype": "float32",
        "feature_spec_hash": spec["feature_spec_hash"],
    }
    (out / "preprocessing.json").write_text(json.dumps(prep, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    fit = fit_threshold(saved["y_true"], saved["probs"], saved["quality_pass"])
    policy = {
        "schema_version": "lsc.thresholds.v1",
        "policy_version": f"maxsoftmax-{_sha(run / 'predictions_val.npz')[:8]}",
        "mode": "max_softmax_threshold",
        "threshold": fit["threshold"],
        "requires_confirmation": True,  # negativos aún no medidos (A4): siempre sugerencia
        "negatives_rejection_measured": False,
        "provisional": True,
        "fitted_on": "validación (procesables) de la ejecución exportada",
        "fit": fit,
        "synthetic": synthetic,
    }
    (out / "thresholds.json").write_text(json.dumps(policy, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    version = f"{'SYNTHETIC-DEMO-' if synthetic else ''}{run_id}-{_sha(out / 'model.keras')[:8]}"
    (out / "model_card.md").write_text(_model_card(metrics, fit, version, synthetic, run_id), encoding="utf-8")

    copied, missing = [], []
    for t in TASKS:
        src = Path(models_dir) / t
        if src.is_file():
            shutil.copy2(src, out / t)
            copied.append(t)
        elif (out / t).is_file():
            copied.append(t)
        else:
            missing.append(t)

    files = ["model.keras", "labels.json", "feature_spec.json", "preprocessing.json", "thresholds.json", "model_card.md", *copied]
    manifest = {
        "schema_version": "lsc.manifest.v1",
        "model_version": version,
        "feature_spec_hash": spec["feature_spec_hash"],
        "run_id": run_id,
        "config_hash": metrics["config_hash"],
        "data_sha256": metrics["data"]["sha256"],
        "evidence_valid": not synthetic,
        "synthetic_demo": synthetic,
        "a4_complete": False,
        "test_used": False,
        "parity_max_abs_diff": max_diff,
        "environment": metrics["environment"],
        "files_sha256": {f: _sha(out / f) for f in files},
        "missing_extractor_assets": missing,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out


def _model_card(m: dict, fit: dict, version: str, synthetic: bool, run_id: str) -> str:
    banner = (
        "> **DEMO SINTÉTICA.** Entrenado con `synthetic_contract_fixture`. No reconoce LSC y no es evidencia de nada.\n\n"
        if synthetic
        else "> Estado: intermedio (A3/A4 parcial). El test NO se ha usado; no hay métricas finales.\n\n"
    )
    val = (
        "No se reportan: datos sintéticos."
        if synthetic
        else f"Validación (procesables): exactitud {m['val']['accuracy_processable']:.3f}, F1 macro {m['val']['macro_f1_processable']:.3f} "
        f"(n={m['val']['n_processable']}). Es validación, no test; la validación es de pocas personas."
    )
    return (
        f"# Tarjeta del modelo — {version}\n\n{banner}"
        f"- **Ejecución:** `{run_id}` ({m['model']}, aumento: {'sí' if m['augment'] else 'no'}, semilla {m['seed']}), {m['n_params']} parámetros.\n"
        "- **Entrada:** (6, 147) float32 del extractor `features_v1`. **Salida:** 10 probabilidades (LSC70W).\n"
        "- **Uso previsto:** una seña aislada por intento, con inicio y fin explícitos. No hay frases ni detección continua.\n"
        f"- **{val}**\n"
        f"- **Umbral provisional:** {fit['threshold']:.4f} (cobertura en validación {fit['coverage']:.2f}, exactitud entre aceptados "
        f"{fit['accuracy_accepted']:.2f}; metas {'cumplidas' if fit['goals_met'] else 'NO cumplidas'}). Rechazo de negativos: sin medir.\n"
        "- **Límites:** la probabilidad softmax no es certeza. Presentar siempre como sugerencia que requiere confirmación.\n"
        "- **Pendiente (A4):** prueba externa única, calibración con webcam de desarrollo, medición de negativos.\n"
    )
