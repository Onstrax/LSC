"""Orquestador de las 12 ejecuciones (A3). Reanudable e idempotente.

Una ejecución se considera completa si existen ``metrics.json`` y ``model.keras``
y su ``config_hash`` coincide con la configuración actual. Si la configuración
cambió, la ejecución se rehace (nunca se mezclan resultados de configs distintas).
"""

from __future__ import annotations

import json
from pathlib import Path

from .config import FAST_OVERRIDES, ConfigError, RunSpec, deep_merge, expand_runs, load_config
from .data import Bundle


def is_complete(spec: RunSpec, runs_dir: Path) -> bool:
    d = runs_dir / spec.run_id
    m = d / "metrics.json"
    if not (m.is_file() and (d / "model.keras").is_file()):
        return False
    try:
        return json.loads(m.read_text(encoding="utf-8")).get("config_hash") == spec.config_hash()
    except json.JSONDecodeError:
        return False


def run_all(
    config_path: str | Path,
    out_dir: str | Path,
    *,
    bundle_path: str | Path | None = None,
    only: list[str] | None = None,
    overrides: dict | None = None,
    force: bool = False,
    fast: bool = False,
) -> list[dict]:
    from .train import train_one_run

    if fast:
        overrides = deep_merge(overrides or {}, FAST_OVERRIDES)
    cfg = load_config(config_path, overrides)
    bpath = Path(bundle_path) if bundle_path else Path(cfg["data"]["bundle_dir"]) / cfg["data"]["features_file"]
    bundle = Bundle(bpath)
    if fast and not bundle.synthetic:
        raise ConfigError("--fast (presupuesto reducido) solo se permite con datos sintéticos, nunca con datos reales")
    runs_dir = Path(out_dir) / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for spec in expand_runs(cfg):
        if only and spec.name not in only and spec.run_id not in only:
            continue
        if not force and is_complete(spec, runs_dir):
            print(f"[skip] {spec.run_id} (completa)")
            results.append(json.loads((runs_dir / spec.run_id / "metrics.json").read_text(encoding="utf-8")))
            continue
        print(f"[run ] {spec.run_id}", flush=True)
        res = train_one_run(spec, bundle, runs_dir)
        print(
            f"       val F1 macro={res['val']['macro_f1_processable']:.3f} "
            f"acc={res['val']['accuracy_processable']:.3f} epoch*={res['best_epoch']} "
            f"p95={res['latency']['p95_ms']:.1f} ms",
            flush=True,
        )
        results.append(res)
    return results
