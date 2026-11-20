"""El modelo por defecto de la app es v2 (artifacts_v2); v1 (congelado) no se toca."""

from __future__ import annotations

from pathlib import Path

import pytest

from api.config import load_settings
from api.constants import DEFAULT_ARTIFACTS_DIR
from src.lsc_models import app_tools, cli

ROOT = Path(__file__).resolve().parent.parent.parent


def test_every_layer_agrees_on_the_default_package(monkeypatch):
    monkeypatch.delenv("LSC_ARTIFACTS_DIR", raising=False)
    assert DEFAULT_ARTIFACTS_DIR == app_tools.DEFAULT_APP_ARTIFACTS == "artifacts_v2"
    assert load_settings().artifacts_dir == Path("artifacts_v2")
    assert "source: ../artifacts_v2" in (ROOT / "deployment" / "compose.yaml").read_text(encoding="utf-8")
    assert 'default="artifacts_v2"' in (ROOT / "deployment" / "make_release.py").read_text(encoding="utf-8")


def test_environment_variable_still_overrides_the_default(monkeypatch):
    monkeypatch.setenv("LSC_ARTIFACTS_DIR", "artifacts")           # volver a v1 sigue siendo posible
    assert load_settings().artifacts_dir == Path("artifacts")


def test_run_app_and_check_app_default_to_v2_and_accept_v1(monkeypatch):
    seen = {}
    monkeypatch.setattr(app_tools, "run_app", lambda mode, artifacts, port, open_browser, debug_dump=None: seen.update(run=artifacts) or 0)
    monkeypatch.setattr(app_tools, "check_app_spawn", lambda mode, artifacts: seen.update(check=artifacts) or 0)
    assert cli.main(["run-app", "--mode", "real"]) == 0 and seen["run"] == "artifacts_v2"
    assert cli.main(["check-app", "--spawn", "real"]) == 0 and seen["check"] == "artifacts_v2"
    assert cli.main(["run-app", "--mode", "real", "--artifacts", "artifacts"]) == 0 and seen["run"] == "artifacts"


def test_v1_commands_keep_pointing_at_v1(monkeypatch):
    """`export-artifacts` y `final-eval` son de v1: sus valores por defecto NO cambian (no se pisa v1 por accidente)."""
    import src.lsc_models.export as export
    import src.lsc_models.final_eval as final_eval

    seen = {}

    def fake_export(reports, out, bundle, **kw):
        seen["export_out"], seen["export_reports"] = str(out), str(reports)
        return Path(out)

    def fake_final(reports, artifacts, bundle, cfg):
        seen["final_artifacts"], seen["final_reports"] = str(artifacts), str(reports)
        raise final_eval.FinalEvalError("no se ejecuta en la prueba")

    monkeypatch.setattr(export, "export_artifacts", fake_export)
    monkeypatch.setattr(final_eval, "run_final_eval", fake_final)
    assert cli.main(["export-artifacts"]) == 0
    assert seen["export_out"] == "artifacts" and seen["export_reports"] == "reports/models"
    assert cli.main(["final-eval"]) == 2                                   # rechazado por el doble, sin abrir el test
    assert seen["final_artifacts"] == "artifacts" and seen["final_reports"] == "reports/models"


def test_missing_default_package_gives_an_actionable_message(tmp_path, capsys):
    assert app_tools.run_app("real", str(tmp_path / "no_hay"), 8000, False) == 3
    err = capsys.readouterr().err
    assert "artifacts_v2" in err and "reports/models_v2" in err
