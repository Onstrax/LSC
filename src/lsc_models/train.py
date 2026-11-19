"""Entrenamiento reproducible de UNA ejecución (configuración + semilla).

- Semillas globales (python, numpy, keras) y operaciones deterministas de TF.
- Entrena solo con muestras procesables de ``train``; el aumento ``train_v1``
  (Ortiz) se aplica por época y solo a train, con semilla derivada de
  (semilla, época, índice).
- Parada temprana y selección de época por F1 macro de validación (procesables);
  desempate por menor pérdida. NUNCA se lee el split ``test``.
- Escribe: config.json, metrics.json, history.csv, predictions_val.npz,
  confusion_val.csv, per_class_val.csv y model.keras.
"""

from __future__ import annotations

import csv
import json
import platform
import random
import time
from pathlib import Path

import numpy as np

from . import metrics as M
from .baseline import MajorityClassifier
from .config import RunSpec
from .data import Bundle


def set_global_determinism(seed: int) -> None:
    import keras
    import tensorflow as tf

    random.seed(seed)
    np.random.seed(seed)
    keras.utils.set_random_seed(seed)  # python + numpy + backend
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:  # pragma: no cover - ya habilitado
        pass


def environment_info() -> dict:
    import keras
    import tensorflow as tf

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "tensorflow": tf.__version__,
        "keras": keras.__version__,
    }


def _aug_seed(seed: int, epoch: int, idx: int) -> int:
    return int(np.random.SeedSequence([seed, epoch, idx]).generate_state(1)[0])


def augment_epoch(X: np.ndarray, seed: int, epoch: int, policy: str) -> np.ndarray:
    from src.lsc_features.augment import augment_train

    return np.stack([augment_train(X[i], _aug_seed(seed, epoch, i), policy) for i in range(len(X))]).astype(np.float32)


def measure_latency(model, x_single: np.ndarray, warmup: int, repeats: int) -> dict:
    """Latencia de inferencia del MODELO con lote 1 (no incluye extractor ni red)."""
    x = x_single[None, ...].astype(np.float32)
    for _ in range(warmup):
        model(x, training=False)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        model(x, training=False).numpy()
        times.append((time.perf_counter() - t0) * 1000.0)
    return {"p50_ms": float(np.percentile(times, 50)), "p95_ms": M.p95(times), "repeats": repeats}


def train_one_run(spec: RunSpec, bundle: Bundle, out_dir: str | Path) -> dict:
    from .architectures import build_model

    out = Path(out_dir) / spec.run_id
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()
    set_global_determinism(spec.seed)

    tr_all = bundle.load_split("train")
    va_all = bundle.load_split("val")  # NUNCA "test" aquí
    tr = tr_all.processable()
    tcfg = spec.training

    import keras

    model = build_model(spec.model, spec.model_params)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=float(tcfg["learning_rate"])),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    best = {"f1": -1.0, "loss": float("inf"), "epoch": 0, "weights": None}
    since_best = 0
    history: list[dict] = []
    for epoch in range(1, int(tcfg["max_epochs"]) + 1):
        X_ep = augment_epoch(tr.X, spec.seed, epoch, spec.augmentation["policy"]) if spec.augment else tr.X
        h = model.fit(X_ep, tr.y, batch_size=int(tcfg["batch_size"]), epochs=1, shuffle=True, verbose=0)
        p_val = model.predict(va_all.X, batch_size=64, verbose=0)
        ev = M.evaluate_split(va_all.y, p_val, va_all.quality_pass)
        f1, vloss = ev["macro_f1_processable"], ev["loss_processable"]
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(h.history["loss"][0]),
                "train_accuracy": float(h.history["accuracy"][0]),
                "val_loss": vloss,
                "val_accuracy": ev["accuracy_processable"],
                "val_macro_f1": f1,
            }
        )
        improved = (f1 > best["f1"] + 1e-12) or (abs(f1 - best["f1"]) <= 1e-12 and vloss < best["loss"])
        if improved:
            best.update(f1=f1, loss=vloss, epoch=epoch, weights=model.get_weights())
            since_best = 0
        else:
            since_best += 1
        if epoch >= int(tcfg["min_epochs"]) and since_best >= int(tcfg["patience"]):
            break

    if tcfg.get("restore_best", True) and best["weights"] is not None:
        model.set_weights(best["weights"])

    # --- evaluación final (train/val) con los pesos elegidos ----------------
    p_val = model.predict(va_all.X, batch_size=64, verbose=0).astype(np.float32)
    p_tr = model.predict(tr.X, batch_size=64, verbose=0)
    val_eval = M.evaluate_split(va_all.y, p_val, va_all.quality_pass)
    train_eval = {
        "n": int(len(tr)),
        "accuracy": M.accuracy(tr.y, np.argmax(p_tr, 1)),
        "macro_f1": M.macro_f1(tr.y, np.argmax(p_tr, 1)),
    }
    pred_val = np.argmax(p_val, axis=1)
    qp = va_all.quality_pass
    per_class = M.per_class_table(va_all.y[qp], pred_val[qp])
    conf = M.confusion(va_all.y[qp], pred_val[qp])

    # --- baseline mayoritario (mismos datos) --------------------------------
    maj = MajorityClassifier().fit(tr.y)
    p_maj = maj.predict_proba(len(va_all))
    baseline_eval = M.evaluate_split(va_all.y, p_maj, va_all.quality_pass)
    baseline_eval["majority_class"] = maj.majority_class

    latency = measure_latency(model, va_all.X[0], int(spec.latency.get("warmup", 20)), int(spec.latency.get("repeats", 200)))

    # --- escritura ----------------------------------------------------------
    (out / "config.json").write_text(json.dumps(spec.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with (out / "history.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(history[0].keys()))
        w.writeheader()
        w.writerows(history)
    np.savez(
        out / "predictions_val.npz",
        sample_id=va_all.sample_id,
        y_true=va_all.y,
        probs=p_val,
        quality_pass=qp,
    )
    np.savetxt(out / "confusion_val.csv", conf, fmt="%d", delimiter=",")
    with (out / "per_class_val.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_class[0].keys()))
        w.writeheader()
        w.writerows(per_class)
    model.save(out / "model.keras")

    result = {
        "run_id": spec.run_id,
        "name": spec.name,
        "seed": spec.seed,
        "model": spec.model,
        "augment": spec.augment,
        "config_hash": spec.config_hash(),
        "data": {**{k: v for k, v in bundle.summary().items() if not k.startswith("n_test")}},
        "evidence_valid": not bundle.synthetic,
        "data_source": "synthetic_contract_fixture" if bundle.synthetic else "features_v1",
        "best_epoch": best["epoch"],
        "epochs_run": len(history),
        "n_params": int(model.count_params()),
        "train": train_eval,
        "val": val_eval,
        "baseline_majority_val": baseline_eval,
        "latency": latency,
        "runtime_s": round(time.perf_counter() - t_start, 2),
        "environment": environment_info(),
        "split_used": ["train", "val"],
    }
    (out / "metrics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result
