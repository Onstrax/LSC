"""``python -m src.lsc_models verify``: comprobación de extremo a extremo.

Imprime una lista PASS / FAIL / SKIP y sale con 0 si no hay ningún FAIL.
Con datos reales ausentes, esos pasos salen SKIP (no PASS): verificar la tubería
con datos sintéticos NO equivale a validar el reconocimiento.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np

REAL_BUNDLE = Path("artifacts/data/v1/features.npz")
PASS, FAIL, SKIP, WARN = "PASS", "FAIL", "SKIP", "WARN"


class _Report:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str]] = []

    def add(self, status: str, name: str, detail: str = "") -> None:
        self.rows.append((status, name, detail))
        mark = {"PASS": "[ OK ]", "FAIL": "[FAIL]", "SKIP": "[SKIP]", "WARN": "[WARN]"}[status]
        print(f"{mark} {name}" + (f" — {detail}" if detail else ""), flush=True)

    def step(self, name: str):
        rep = self

        class _Ctx:
            def __enter__(self_inner):
                self_inner.t0 = time.time()
                return self_inner

            def __exit__(self_inner, et, ev, tb):
                if et is None:
                    return False
                rep.add(FAIL, name, f"{et.__name__}: {ev}")
                if "--trace" in sys.argv:
                    traceback.print_exception(et, ev, tb)
                return True  # no abortar: seguir con el resto

        return _Ctx()

    @property
    def ok(self) -> bool:
        return not any(s == FAIL for s, _, _ in self.rows)


def run_verify(with_tests: bool = False) -> int:
    rep = _Report()
    print("Verificación del proyecto LSC (módulo de modelos)\n")

    # 1 ---------------------------------------------------------------
    with rep.step("1. Dependencias importables"):
        import keras
        import matplotlib
        import sklearn
        import tensorflow as tf
        import yaml  # noqa: F401

        rep.add(PASS, "1. Dependencias importables",
                f"python {sys.version.split()[0]}, tf {tf.__version__}, keras {keras.__version__}, "
                f"numpy {np.__version__}, sklearn {sklearn.__version__}, matplotlib {matplotlib.__version__}")

    # 2 ---------------------------------------------------------------
    with rep.step("2. Configuración de experimentos (4 configs x 3 semillas = 12)"):
        from .config import expand_runs, load_config

        runs = expand_runs(load_config())
        assert len(runs) == 12, len(runs)
        rep.add(PASS, "2. Configuración de experimentos (4 configs x 3 semillas = 12)", "12 ejecuciones, semillas 17/29/43")

    # 3 ---------------------------------------------------------------
    with rep.step("3. Contratos de Ortiz (etiquetas y feature_spec_hash)"):
        from contracts.canonical import feature_spec_hash

        spec = json.loads(Path("contracts/feature_spec.json").read_text(encoding="utf-8"))
        assert feature_spec_hash(spec) == spec["feature_spec_hash"], "feature_spec_hash no coincide"
        labels = json.loads(Path("contracts/labels.json").read_text(encoding="utf-8"))["labels"]
        assert [l["class_id"] for l in labels] == list(range(10))
        rep.add(PASS, "3. Contratos de Ortiz (etiquetas y feature_spec_hash)", f"hash {spec['feature_spec_hash'][:12]}…, 10 clases")
        from .prepare import contract_warnings

        for w in contract_warnings(spec):
            rep.add(WARN, "3b. Defecto en el contrato de Ortiz (no bloquea)", w)

    # 4 ---------------------------------------------------------------
    real_ok = False
    if REAL_BUNDLE.is_file():
        with rep.step("4. features.npz REAL válido"):
            from .data import Bundle

            s = Bundle(REAL_BUNDLE).summary()
            assert not s["synthetic"], "artifacts/data/v1 contiene un bundle sintético"
            real_ok = True
            rep.add(PASS, "4. features.npz REAL válido", f"N={s['n_total']}, train/val/test={s['n_train']}/{s['n_val']}/{s['n_test']}")
    else:
        rep.add(SKIP, "4. features.npz REAL válido",
                "no existe artifacts/data/v1/features.npz -> `python -m src.lsc_models prepare-data`")

    # 5 ---------------------------------------------------------------
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        syn = None
        with rep.step("5. Bundle sintético (solo tubería; NO es evidencia)"):
            from .data import Bundle, make_synthetic_contract_fixture

            syn = make_synthetic_contract_fixture(tmp / "syn" / "features.npz", n_signers=20)
            assert Bundle(syn).synthetic
            rep.add(PASS, "5. Bundle sintético (solo tubería; NO es evidencia)", "forma (N,6,147) float32, sin fuga entre personas")
        out = tmp / "out"

        if syn is not None:
            with rep.step("6. Las 12 ejecuciones (A1–A3) con presupuesto reducido"):
                from .experiments import run_all

                t0 = time.time()
                res = run_all("configs/train/experiments.yaml", out, bundle_path=syn, fast=True)
                assert len(res) == 12
                assert all(r["split_used"] == ["train", "val"] for r in res), "una ejecución usó otro split"
                rep.add(PASS, "6. Las 12 ejecuciones (A1–A3) con presupuesto reducido", f"{time.time() - t0:.0f} s; test no leído")

            with rep.step("7. Tablas, selección y gráficas"):
                from .ablation import load_runs, make_plots, write_tables
                from .config import load_config

                sel = write_tables(load_runs(out / "runs"), out, load_config())
                figs = make_plots(out)
                assert sel["evidence_valid"] is False and sel["goals_vs_validation"]["evaluated"] is False
                assert sel["deployed_run"].endswith("__seed17")
                rep.add(PASS, "7. Tablas, selección y gráficas", f"{sel['deployed_run']}; {len(figs)} figuras; metas NO evaluadas (sintético)")

            with rep.step("8. Sensibilidad a fallos del extractor"):
                from .config import load_config
                from .data import Bundle
                from .sensitivity import run_sensitivity

                p = run_sensitivity(Bundle(syn), out / "runs", out, load_config(), fast=True)
                assert p.is_file() and (out / "quality_rejections_val.csv").is_file()
                rep.add(PASS, "8. Sensibilidad a fallos del extractor", p.name)

            with rep.step("9. Test reservado: bloqueado y congelación rechazada con sintéticos"):
                from .ablation import write_freeze
                from .config import load_config
                from .data import Bundle, TestAccessError

                b = Bundle(syn, freeze_file=tmp / "no_freeze.json")
                for kwargs in ({}, {"allow_test": True}):
                    try:
                        b.load_split("test", **kwargs)
                    except TestAccessError:
                        pass
                    else:
                        raise AssertionError("el split de test se pudo leer sin congelar")
                try:
                    write_freeze(out, load_config())
                except RuntimeError:
                    pass
                else:
                    raise AssertionError("freeze aceptó ejecuciones sintéticas")
                rep.add(PASS, "9. Test reservado: bloqueado y congelación rechazada con sintéticos")

            with rep.step("10. Predictor exportado == modelo en memoria (tolerancia 1e-5)"):
                import keras

                from src.lsc_inference import Predictor

                from . import custom_layers  # noqa: F401

                mp = out / "runs" / "transformer_aug__seed17" / "model.keras"
                x = np.random.default_rng(0).normal(size=(4, 6, 147)).astype(np.float32)
                ref = keras.models.load_model(mp, compile=False).predict(x, verbose=0)
                got = Predictor(mp).predict(x)
                assert got.shape == (4, 10) and np.allclose(got, ref, atol=1e-5)
                rep.add(PASS, "10. Predictor exportado == modelo en memoria (tolerancia 1e-5)", f"máx |Δ| = {np.abs(got - ref).max():.2e}")

            with rep.step("11. Integración con la API de Juan (RealPredictor + RealDecisionAdapter)"):
                import shutil

                from api.adapters.real import RealDecisionAdapter, RealPredictor

                art = tmp / "artifacts"
                art.mkdir()
                shutil.copy(out / "runs" / "bilstm_noaug__seed17" / "model.keras", art / "model.keras")
                (art / "manifest.json").write_text(json.dumps({"model_version": "verify", "feature_spec_hash": "x"}))
                (art / "thresholds.json").write_text(json.dumps({"policy_version": "verify", "threshold": 0.0}))
                probs = RealPredictor(art).predict(x[:1])
                assert RealDecisionAdapter(art).decide(probs).accepted
                rep.add(PASS, "11. Integración con la API de Juan (RealPredictor + RealDecisionAdapter)")

            with rep.step("12. export-artifacts (paquete DEMO sintético) con paridad y umbral"):
                from .export import export_artifacts

                pkg = export_artifacts(out, tmp / "pkg", syn, run_id="bilstm_noaug__seed17", allow_synthetic=True, models_dir=tmp / "none")
                man = json.loads((pkg / "manifest.json").read_text())
                assert man["synthetic_demo"] and man["parity_max_abs_diff"] <= 1e-5
                rep.add(PASS, "12. export-artifacts (paquete DEMO sintético) con paridad y umbral", f"{man['model_version']}; paridad {man['parity_max_abs_diff']:.1e}")

            with rep.step("13. App completa en modo simulado por HTTP (10 comprobaciones)"):
                from .app_tools import check_app_spawn

                assert check_app_spawn("mock", "artifacts") == 0, "check-app falló (ejecute `check-app --spawn mock` para el detalle)"
                rep.add(PASS, "13. App completa en modo simulado por HTTP (10 comprobaciones)", "API + interfaz + /predict con 6 JPEG + errores")

    # 12 --------------------------------------------------------------
    if with_tests:
        with rep.step("14. pytest tests/models tests/inference"):
            r = subprocess.run([sys.executable, "-m", "pytest", "tests/models", "tests/inference", "-q", "-p", "no:cacheprovider"],
                               capture_output=True, text=True)
            tail = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-200:]
            assert r.returncode == 0, tail
            rep.add(PASS, "14. pytest tests/models tests/inference", tail)
    else:
        rep.add(SKIP, "14. pytest tests/models tests/inference", "use `verify --with-tests` o ejecute pytest a mano")

    n_fail = sum(1 for s, _, _ in rep.rows if s == FAIL)
    n_skip = sum(1 for s, _, _ in rep.rows if s == SKIP)
    n_warn = sum(1 for s, _, _ in rep.rows if s == WARN)
    print("\n" + "=" * 70)
    if n_fail:
        print(f"RESULTADO: {n_fail} paso(s) FALLARON. Reejecute con `verify --trace` para ver el detalle.")
        return 1
    print(f"RESULTADO: la tubería funciona ({len(rep.rows) - n_skip - n_warn} pasos OK, {n_skip} omitidos, {n_warn} avisos).")
    if not real_ok:
        print("AVISO: faltan los datos REALES. Esto prueba el código, no el reconocimiento de LSC.\n"
              "       Siguiente paso: `python -m src.lsc_models prepare-data`.")
    return 0
