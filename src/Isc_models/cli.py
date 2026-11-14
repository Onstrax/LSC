"""CLI: ``python -m src.lsc_models <comando>``.

  synthetic-fixture  genera el bundle sintético (SOLO pruebas de interfaz)
  run-all            las 12 ejecuciones (reanudable)
  aggregate          tablas + selección + gráficas desde lo guardado
  sensitivity        sensibilidad a fallos del extractor (validación)
  freeze             congela decisiones (habilita el test único de A4)

Códigos de salida: 0 éxito, 2 configuración/entrada inválida, 3 datos ausentes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import ConfigError, DEFAULT_CONFIG, load_config

DEFAULT_OUT = "reports/models"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m src.lsc_models")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("synthetic-fixture")
    s.add_argument("--out", required=True)

    for name in ("run-all", "aggregate", "sensitivity", "freeze"):
        p = sub.add_parser(name)
        p.add_argument("--config", default=str(DEFAULT_CONFIG))
        p.add_argument("--out", default=DEFAULT_OUT)
        if name in ("run-all", "sensitivity"):
            p.add_argument("--bundle", default=None, help="ruta a features.npz (por defecto, la de la configuración)")
        if name == "run-all":
            p.add_argument("--only", nargs="*", default=None)
            p.add_argument("--force", action="store_true")

    a = ap.parse_args(argv)
    try:
        if a.cmd == "synthetic-fixture":
            from .data import make_synthetic_contract_fixture

            print(make_synthetic_contract_fixture(a.out))
            return 0
        cfg = load_config(a.config)
        if a.cmd == "run-all":
            from .experiments import run_all

            run_all(a.config, a.out, bundle_path=a.bundle, only=a.only, force=a.force)
        elif a.cmd == "aggregate":
            from .ablation import load_runs, make_plots, write_tables

            runs = load_runs(Path(a.out) / "runs")
            sel = write_tables(runs, a.out, cfg)
            make_plots(a.out)
            print(f"selección: {sel['chosen_config']} -> {sel['deployed_run']} ({sel['reason']})")
        elif a.cmd == "sensitivity":
            from .data import Bundle
            from .sensitivity import run_sensitivity

            bpath = a.bundle or Path(cfg["data"]["bundle_dir"]) / cfg["data"]["features_file"]
            print(run_sensitivity(Bundle(bpath), Path(a.out) / "runs", a.out, cfg))
        elif a.cmd == "freeze":
            from .ablation import write_freeze

            print(write_freeze(a.out, cfg))
        return 0
    except ConfigError as exc:
        print(f"config inválida: {exc}", file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print(f"falta un archivo: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
