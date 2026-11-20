"""CLI: ``python -m src.lsc_models <comando>``.

  prepare-data       genera features.npz: REAL (pipeline de Ortiz) o --synthetic
  download-models    descarga los .task oficiales y los verifica contra el contrato
  final-eval         evaluación ÚNICA sobre el test (tras `freeze` y `export-artifacts`)
  sync-bundle-spec   alinea el bundle ya generado con el contrato corregido (sin reprocesar)
  export-artifacts   arma artifacts/ (modelo, política, manifest) para la app en modo real
  run-app            arranca la aplicación (--mode mock|real)
  check-app          prueba por HTTP una app en marcha (o --spawn mock|real)
  diagnose-dump      analiza los intentos guardados con `run-app --debug-dump` (qué ve el modelo en vivo)
  robustness         compara v1 y v2 en validación perturbada (mano ausente, recorte); NO abre el test
  authorize-test     autoriza (con constancia) un SEGUNDO uso del test para v2; sin esto v2 sigue sellado
  compare-test       compara v1 y v2 de forma pareada con las predicciones de test YA guardadas (no reevalúa)
  verify             comprueba que todo el proyecto funciona (PASS/FAIL)
  synthetic-fixture  genera el bundle sintético (SOLO pruebas de interfaz)
  run-all            las 12 ejecuciones (reanudable)
  aggregate          tablas + selección + gráficas desde lo guardado
  sensitivity        sensibilidad a fallos del extractor (validación)
  freeze             congela decisiones (habilita el test único de A4)

Códigos de salida: 0 éxito, 2 configuración/entrada inválida, 3 datos ausentes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .app_tools import DEFAULT_APP_ARTIFACTS
from .config import ConfigError, DEFAULT_CONFIG, load_config

DEFAULT_OUT = "reports/models"


def main(argv: list[str] | None = None) -> int:
    import os

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # silencia avisos C++ inofensivos de TensorFlow
    ap = argparse.ArgumentParser(prog="python -m src.lsc_models")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("synthetic-fixture")
    s.add_argument("--out", required=True)

    pd = sub.add_parser("prepare-data")
    pd.add_argument("--raw-dir", default=None, help="carpeta LSC70W del corpus crudo")
    pd.add_argument("--models-dir", default=None, help="carpeta con los .task de MediaPipe")
    pd.add_argument("--out", default=None)
    pd.add_argument("--no-download", action="store_true", help="no intentar descargar los .task")
    pd.add_argument("--accept-mismatch", action="store_true", help="aceptar .task que no coinciden con el contrato (rompe procedencia)")
    pd.add_argument("--synthetic", action="store_true", help="bundle sintético (NO es evidencia)")

    fe = sub.add_parser("final-eval")
    fe.add_argument("--config", default=str(DEFAULT_CONFIG))
    fe.add_argument("--reports", default=DEFAULT_OUT)
    fe.add_argument("--artifacts", default="artifacts")
    fe.add_argument("--bundle", default=None)

    sb = sub.add_parser("sync-bundle-spec")
    sb.add_argument("--bundle-dir", default=None, help="por defecto artifacts/data/v1")

    ea = sub.add_parser("export-artifacts")
    ea.add_argument("--reports", default=DEFAULT_OUT)
    ea.add_argument("--out", default="artifacts")
    ea.add_argument("--bundle", default=None)
    ea.add_argument("--run", default=None, help="por defecto, la ejecución desplegable de selection.json")
    ea.add_argument("--allow-synthetic", action="store_true", help="DEMO de integración con datos sintéticos")
    ea.add_argument("--force", action="store_true")

    ra = sub.add_parser("run-app")
    ra.add_argument("--mode", choices=["mock", "real"], default="mock")
    ra.add_argument("--artifacts", default=DEFAULT_APP_ARTIFACTS, help="paquete del modelo; por defecto v2 (v1: --artifacts artifacts)")
    ra.add_argument("--port", type=int, default=8000)
    ra.add_argument("--open", action="store_true", help="abrir el navegador")
    ra.add_argument("--debug-dump", nargs="?", const="artifacts/debug_dump", default=None, metavar="DIR",
                    help="guarda localmente cada intento (6 cuadros CON ROSTRO, x y probabilidades) para diagnóstico; "
                         "por defecto artifacts/debug_dump (ignorada por Git)")

    at = sub.add_parser("authorize-test")
    at.add_argument("--reports", default="reports/models_v2", help="carpeta de resultados de v2")
    at.add_argument("--reason", required=True, help="motivo; queda registrado en test_authorization.json")
    at.add_argument("--confirm-second-use", action="store_true",
                    help="confirma que es un SEGUNDO uso del split de test (el primero fue con v1)")

    ct = sub.add_parser("compare-test")
    ct.add_argument("--v1-reports", default=DEFAULT_OUT)
    ct.add_argument("--v2-reports", default="reports/models_v2")

    rb = sub.add_parser("robustness")
    rb.add_argument("--v1-reports", default=DEFAULT_OUT, help="resultados de v1 (solo lectura)")
    rb.add_argument("--v2-reports", default="reports/models_v2")
    rb.add_argument("--bundle", default=None, help="features.npz (por defecto, el de la configuración)")
    rb.add_argument("--out", default=None, help="CSV de salida (por defecto <v2-reports>/robustness_comparison.csv)")
    rb.add_argument("--only", nargs="*", default=None, help="limitar a estas configuraciones (p. ej. transformer_aug transformer_robust)")

    dd = sub.add_parser("diagnose-dump")
    dd.add_argument("--dir", default="artifacts/debug_dump")
    dd.add_argument("--artifacts", default=DEFAULT_APP_ARTIFACTS)
    dd.add_argument("--expect", default=None, help='seña esperada por intento, en orden: "Hola*3,Buenos*3"')
    dd.add_argument("--no-redetect", action="store_true", help="no volver a correr MediaPipe sobre los cuadros")
    dd.add_argument("--bundle", default=None)
    dd.add_argument("--raw-dir", default=None, help="corpus LSC70W, para medir el tamaño de hombros de referencia")
    dd.add_argument("--corpus-sample", type=int, default=30)

    ca = sub.add_parser("check-app")
    ca.add_argument("--url", default="http://127.0.0.1:8000")
    ca.add_argument("--spawn", choices=["mock", "real"], default=None, help="arrancar un servidor temporal y comprobarlo")
    ca.add_argument("--artifacts", default=DEFAULT_APP_ARTIFACTS)

    dm = sub.add_parser("download-models")
    dm.add_argument("--models-dir", default=None)
    dm.add_argument("--accept-mismatch", action="store_true")

    vf = sub.add_parser("verify")
    vf.add_argument("--trace", action="store_true", help="muestra el traceback de los fallos")
    vf.add_argument("--with-tests", action="store_true", help="además corre pytest de modelos/inferencia")

    for name in ("run-all", "aggregate", "sensitivity", "freeze"):
        p = sub.add_parser(name)
        p.add_argument("--config", default=str(DEFAULT_CONFIG))
        p.add_argument("--out", default=DEFAULT_OUT)
        if name in ("run-all", "sensitivity"):
            p.add_argument("--bundle", default=None, help="ruta a features.npz (por defecto, la de la configuración)")
            p.add_argument("--fast", action="store_true", help="presupuesto reducido; SOLO con datos sintéticos")
        if name == "run-all":
            p.add_argument("--only", nargs="*", default=None)
            p.add_argument("--force", action="store_true")

    a = ap.parse_args(argv)
    try:
        if a.cmd == "prepare-data":
            from .prepare import main_prepare

            return main_prepare(a.raw_dir, a.models_dir, a.out, a.synthetic,
                                download=not a.no_download, accept_mismatch=a.accept_mismatch)
        if a.cmd == "final-eval":
            from .data import TestAccessError
            from .final_eval import FinalEvalError, run_final_eval

            cfg = load_config(a.config)
            bpath = a.bundle or Path(cfg["data"]["bundle_dir"]) / cfg["data"]["features_file"]
            try:
                out = run_final_eval(a.reports, a.artifacts, bpath, cfg)
            except (FinalEvalError, TestAccessError) as exc:
                print(f"evaluación de test rechazada (el test NO se abrió): {exc}", file=sys.stderr)
                return 2
            res = json.loads(Path(out).read_text(encoding="utf-8"))
            d, g = res["deployed"], res["goals_vs_deployed"]
            print(f"TEST (única evaluación) -> {out}\n"
                  f"  desplegado {res['deployed_run']}: exactitud {d['accuracy_processable']:.3f}, F1 macro {d['macro_f1_processable']:.3f} "
                  f"(integral {d['accuracy_integral']:.3f})\n"
                  f"  metas: exactitud>=0.85 {'CUMPLIDA' if g['accuracy']['met'] else 'NO cumplida'}; F1>=0.80 {'CUMPLIDA' if g['macro_f1']['met'] else 'NO cumplida'}\n"
                  "  NO repita esta evaluación: invalidaría el test.")
            return 0
        if a.cmd == "sync-bundle-spec":
            from .migrate import main_sync

            return main_sync(a.bundle_dir)
        if a.cmd == "export-artifacts":
            from .export import ExportError, export_artifacts

            cfg0 = load_config(DEFAULT_CONFIG)
            bpath = a.bundle or Path(cfg0["data"]["bundle_dir"]) / cfg0["data"]["features_file"]
            try:
                print(export_artifacts(a.reports, a.out, bpath, run_id=a.run, allow_synthetic=a.allow_synthetic, force=a.force))
            except ExportError as exc:
                print(f"exportación rechazada: {exc}", file=sys.stderr)
                return 2
            return 0
        if a.cmd == "run-app":
            from .app_tools import run_app

            return run_app(a.mode, a.artifacts, a.port, a.open, debug_dump=a.debug_dump)
        if a.cmd == "authorize-test":
            from .test_authorization import TestAuthorizationError, authorize_second_test_use

            try:
                p = authorize_second_test_use(a.reports, a.reason, confirm=a.confirm_second_use)
            except TestAuthorizationError as exc:
                print(f"autorización rechazada: {exc}", file=sys.stderr)
                return 2
            print(f"{p}: SEGUNDO uso del test autorizado para v2. Siguiente: `freeze` y `final-eval` (una sola vez) con la config v2.")
            return 0
        if a.cmd == "compare-test":
            from .paired import write_paired_comparison

            print(write_paired_comparison(a.v1_reports, a.v2_reports))  # lee predicciones ya guardadas; no reevalúa
            return 0
        if a.cmd == "robustness":
            from .data import Bundle
            from .robustness import compare, summarize_rows

            cfg0 = load_config(DEFAULT_CONFIG)
            bpath = a.bundle or Path(cfg0["data"]["bundle_dir"]) / cfg0["data"]["features_file"]
            va = Bundle(bpath).load_split("val").processable()  # SOLO validación; el test no se abre
            groups: dict[str, list[Path]] = {}
            for tag, rep in (("v1", a.v1_reports), ("v2", a.v2_reports)):
                for d in sorted((Path(rep) / "runs").glob("*__seed*")):
                    name = d.name.split("__")[0]
                    if (d / "model.keras").is_file() and (not a.only or name in a.only):
                        groups.setdefault(f"{tag}:{name}", []).append(d)
            if not groups:
                print("No hay ejecuciones para comparar (¿ya entrenó v2?).", file=sys.stderr)
                return 3
            out_csv = Path(a.out) if a.out else Path(a.v2_reports) / "robustness_comparison.csv"
            rows = compare(groups, va.X, va.y, out_csv)
            summ = summarize_rows(rows)
            cols = ["original", "sin_mano_izq", "sin_mano_der", "recorte_1.0", "recorte_0.7", "recorte_0.5",
                    "manos_por_tramos", "escala_traslacion", "webcam_like", "MEDIA_ROBUSTA"]
            print(f"Exactitud en validación procesable (n={len(va)}); media sobre semillas. CSV: {out_csv}")
            print(f"{'grupo':26s} " + " ".join(f"{c[:11]:>11s}" for c in cols))
            for g in groups:
                vals = {r["escenario"]: r for r in summ if r["grupo"] == g}
                print(f"{g:26s} " + " ".join(f"{vals[c]['accuracy']:>11.3f}" if c in vals else f"{'-':>11s}" for c in cols))
            print("\nConfianza mediana (misma tabla):")
            for g in groups:
                vals = {r["escenario"]: r for r in summ if r["grupo"] == g}
                print(f"{g:26s} " + " ".join(f"{vals[c]['conf_median']:>11.3f}" if c in vals else f"{'-':>11s}" for c in cols))
            return 0
        if a.cmd == "diagnose-dump":
            from .diagnose import run_diagnose

            return run_diagnose(a.dir, a.artifacts, a.expect, redetect=not a.no_redetect, bundle=a.bundle,
                                raw_dir=a.raw_dir, corpus_sample=a.corpus_sample)
        if a.cmd == "check-app":
            from .app_tools import check_app, check_app_spawn

            return check_app_spawn(a.spawn, a.artifacts) if a.spawn else check_app(a.url)
        if a.cmd == "download-models":
            from .prepare import main_download

            return main_download(a.models_dir, a.accept_mismatch)
        if a.cmd == "verify":
            from .verify import run_verify

            return run_verify(with_tests=a.with_tests)
        if a.cmd == "synthetic-fixture":
            from .data import make_synthetic_contract_fixture

            print(make_synthetic_contract_fixture(a.out))
            return 0
        cfg = load_config(a.config)
        if a.cmd == "run-all":
            from .experiments import run_all

            run_all(a.config, a.out, bundle_path=a.bundle, only=a.only, force=a.force, fast=a.fast)
        elif a.cmd == "aggregate":
            from .ablation import load_runs, make_plots, write_tables
            from .config import config_names

            names = config_names(cfg)
            runs = load_runs(Path(a.out) / "runs", names)
            sel = write_tables(runs, a.out, cfg)
            make_plots(a.out, names)
            print(f"selección: {sel['chosen_config']} -> {sel['deployed_run']} ({sel['reason']})")
        elif a.cmd == "sensitivity":
            from .config import config_names
            from .data import Bundle
            from .sensitivity import run_sensitivity

            bpath = a.bundle or Path(cfg["data"]["bundle_dir"]) / cfg["data"]["features_file"]
            print(run_sensitivity(Bundle(bpath), Path(a.out) / "runs", a.out, cfg, fast=a.fast, names=config_names(cfg)))
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
