"""Evaluación FINAL sobre el test (A4). Se ejecuta UNA sola vez.

Precondiciones (si alguna falla, el test no se abre):
- existe ``reports/models/freeze.json`` (decisiones congeladas; lo crea ``freeze``);
- los 3 modelos de la configuración elegida y el bundle son EXACTAMENTE los congelados (SHA-256);
- existe el paquete exportado (``export-artifacts``) con ``thresholds.json`` de la ejecución desplegable;
- no existe ``test_evaluation.json`` (la evaluación única no se repite; repetirla invalidaría el test).

Se informan por separado (plan §11): exactitud sobre procesables, desempeño integral (rechazos de calidad
cuentan como no acertados) y aceptación por umbral. El rechazo de NEGATIVOS no se mide aquí (requiere
capturas de negativos) y se declara como no medido. Las metas se comparan, no se presentan como resultados.
"""

from __future__ import annotations

import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from . import metrics as M
from .baseline import MajorityClassifier
from .config import REQUIRED_SEEDS
from .data import Bundle


class FinalEvalError(RuntimeError):
    pass


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def acceptance_stats(y: np.ndarray, probs: np.ndarray, quality_pass: np.ndarray, threshold: float) -> dict:
    """Cobertura y exactitud entre aceptados con el umbral de máximo-softmax (>=, como DecisionPolicy)."""
    qp = np.asarray(quality_pass, dtype=bool)
    yp, pp = np.asarray(y)[qp], np.asarray(probs)[qp]
    conf, pred = pp.max(axis=1), pp.argmax(axis=1)
    acc_mask = conf >= threshold
    n_acc = int(acc_mask.sum())
    return {
        "threshold": float(threshold),
        "n_processable": int(len(yp)),
        "n_accepted": n_acc,
        "coverage_processable": float(acc_mask.mean()) if len(yp) else float("nan"),
        "accuracy_among_accepted": float((pred[acc_mask] == yp[acc_mask]).mean()) if n_acc else float("nan"),
        "accepted_over_all_samples": float(n_acc / len(qp)) if len(qp) else float("nan"),
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def run_final_eval(reports_dir, artifacts_dir, bundle_path, cfg: dict, *, allow_synthetic: bool = False) -> Path:
    import keras

    from . import custom_layers  # noqa: F401

    reports, art = Path(reports_dir), Path(artifacts_dir)
    out_json = reports / "test_evaluation.json"
    if out_json.exists():
        raise FinalEvalError(f"{out_json} ya existe: el test se evalúa UNA sola vez. Repetirlo invalida la evaluación intacta.")
    freeze_p = reports / "freeze.json"
    if not freeze_p.is_file():
        raise FinalEvalError("falta freeze.json: congele las decisiones (`freeze`) antes de abrir el test")
    freeze = json.loads(freeze_p.read_text(encoding="utf-8"))
    sel = json.loads((reports / "selection.json").read_text(encoding="utf-8"))
    if not sel.get("evidence_valid", False) and not allow_synthetic:
        raise FinalEvalError("las ejecuciones son sintéticas: no se evalúa el test con ellas")
    chosen, deployed = freeze["chosen_config"], freeze["deployed_run"]

    # --- integridad: modelos y datos congelados --------------------------------
    runs = [f"{chosen}__seed{s}" for s in REQUIRED_SEEDS]
    for rid in runs:
        mp = reports / "runs" / rid / "model.keras"
        if not mp.is_file() or _sha(mp) != freeze["model_sha256"][rid]:
            raise FinalEvalError(f"el modelo {rid} no coincide con el congelado (SHA-256): no se abre el test")
        m = json.loads((reports / "runs" / rid / "metrics.json").read_text(encoding="utf-8"))
        if m["config_hash"] != freeze["config_hashes"][rid]:
            raise FinalEvalError(f"la configuración de {rid} cambió después de congelar")
    bundle = Bundle(bundle_path, freeze_file=freeze_p)
    if [bundle.sha256] != freeze["data_sha256"]:
        raise FinalEvalError("el bundle de datos no es el congelado (SHA-256 distinto)")

    # --- paquete exportado y umbral --------------------------------------------
    man_p, thr_p = art / "manifest.json", art / "thresholds.json"
    if not (man_p.is_file() and thr_p.is_file()):
        raise FinalEvalError(f"falta el paquete en {art}: ejecute `export-artifacts` antes")
    manifest = json.loads(man_p.read_text(encoding="utf-8"))
    if manifest["run_id"] != deployed:
        raise FinalEvalError(f"el paquete exportado es de {manifest['run_id']}, no de la ejecución desplegable {deployed}")
    if manifest.get("test_used"):
        raise FinalEvalError("el manifest ya indica que el test fue usado")
    threshold = float(json.loads(thr_p.read_text(encoding="utf-8"))["threshold"])

    # --- ÚNICA lectura del test -------------------------------------------------
    test = bundle.load_split("test", allow_test=True)
    train_p = bundle.load_split("train").processable()
    per_seed, deployed_detail = [], None
    for rid in runs:
        model = keras.models.load_model(reports / "runs" / rid / "model.keras", compile=False)
        probs = model.predict(test.X, batch_size=64, verbose=0).astype(np.float32)
        ev = M.evaluate_split(test.y, probs, test.quality_pass)
        row = {"run_id": rid, "seed": int(rid.rsplit("seed", 1)[1]), **ev}
        per_seed.append(row)
        if rid == deployed:
            qp = test.quality_pass
            pred = probs.argmax(1)
            np.savez(reports / "predictions_test_deployed.npz", sample_id=test.sample_id, y_true=test.y, probs=probs, quality_pass=qp)
            np.savetxt(reports / "confusion_test_deployed.csv", M.confusion(test.y[qp], pred[qp]), fmt="%d", delimiter=",")
            pc = M.per_class_table(test.y[qp], pred[qp])
            _write_csv(reports / "per_class_test_deployed.csv", pc)
            deployed_detail = {"per_class_file": "per_class_test_deployed.csv", "acceptance": acceptance_stats(test.y, probs, qp, threshold)}

    maj = MajorityClassifier().fit(train_p.y)
    baseline = M.evaluate_split(test.y, maj.predict_proba(len(test)), test.quality_pass)

    def ms(key):
        v = np.array([r[key] for r in per_seed], dtype=np.float64)
        return {"mean": float(v.mean()), "std": float(v.std(ddof=1)), "min": float(v.min()), "max": float(v.max())}

    dep = next(r for r in per_seed if r["run_id"] == deployed)
    goals = cfg["goals"]
    result = {
        "schema_version": "lsc.test_evaluation.v1",
        "evaluated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "evidence_valid": bool(sel.get("evidence_valid", False)),
        "chosen_config": chosen,
        "deployed_run": deployed,
        "freeze_sha256": _sha(freeze_p),
        "data_sha256": bundle.sha256,
        "test_counts": {"n": len(test), "n_processable": int(test.quality_pass.sum())},
        "per_seed": per_seed,
        "summary_over_seeds": {k: ms(k) for k in ("accuracy_processable", "macro_f1_processable", "accuracy_integral", "macro_f1_integral")},
        "deployed": {**dep, **deployed_detail},
        "baseline_majority": baseline,
        "goals_vs_deployed": {
            "scope": "test, muestras procesables, modelo desplegado (semilla 17); la latencia p95 de extremo a extremo la mide J4",
            "accuracy": {"value": dep["accuracy_processable"], "goal": goals["accuracy"], "met": bool(dep["accuracy_processable"] >= goals["accuracy"])},
            "macro_f1": {"value": dep["macro_f1_processable"], "goal": goals["macro_f1"], "met": bool(dep["macro_f1_processable"] >= goals["macro_f1"])},
        },
        "negatives_rejection": "NO MEDIDO (requiere capturas de negativos); la interfaz debe presentar sugerencias a confirmar",
        "test_used_once": True,
    }
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # --- el paquete refleja que el test ya se usó ------------------------------------
    manifest["test_used"] = True
    manifest["test_evaluation_sha256"] = _sha(out_json)
    card = art / "model_card.md"
    d = result["deployed"]
    card.write_text(
        card.read_text(encoding="utf-8")
        + "\n## Resultados en test (evaluación única)\n\n"
        + f"- Modelo desplegado `{deployed}`: exactitud {d['accuracy_processable']:.3f}, F1 macro {d['macro_f1_processable']:.3f} "
        + f"sobre {d['n_processable']} secuencias procesables; integral (rechazos de calidad cuentan como error): "
        + f"exactitud {d['accuracy_integral']:.3f}.\n"
        + f"- Aceptación con umbral {threshold:.4f}: cobertura {d['acceptance']['coverage_processable']:.3f}, "
        + f"exactitud entre aceptados {d['acceptance']['accuracy_among_accepted']:.3f}.\n"
        + "- Rechazo de negativos: no medido.\n",
        encoding="utf-8",
    )
    manifest["files_sha256"]["model_card.md"] = _sha(card)
    man_p.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out_json
