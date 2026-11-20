"""Autorización explícita para un SEGUNDO uso del split de test (modelo v2).

El plan fija que el test se abre UNA sola vez (v1, ya hecho). v2 nació sellado (`evaluation.test_sealed: true`):
`freeze` lo rechaza y sin `freeze.json` `final-eval` no puede abrir el test. El integrador puede decidir abrirlo
para v2 igualmente; no se hace editando la configuración (cambiaría los `config_hash` de v2 y obligaría a
reentrenar) sino con este archivo, que deja constancia verificable de que:

- es el **segundo** uso del split y por qué se autorizó;
- quedó atado al `test_evaluation.json` de v1 vigente (si ese archivo cambia, la autorización deja de valer);
- se declara qué información del test se conocía al diseñar v2.

Sin este archivo, o con uno que no corresponde, el test sigue sellado.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

SCHEMA = "lsc.test_authorization.v1"
V1_TEST_EVALUATION = Path("reports/models/test_evaluation.json")
FILENAME = "test_authorization.json"


class TestAuthorizationError(RuntimeError):
    __test__ = False  # no es una clase de pruebas de pytest


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def authorize_second_test_use(reports_dir: str | Path, reason: str, *, confirm: bool,
                              v1_test_evaluation: str | Path = V1_TEST_EVALUATION) -> Path:
    """Escribe ``test_authorization.json`` en ``reports_dir`` (carpeta de v2)."""
    if not confirm:
        raise TestAuthorizationError("falta la confirmación explícita (--confirm-second-use): es un SEGUNDO uso del test")
    if not reason or len(reason.strip()) < 15:
        raise TestAuthorizationError("indique el motivo (mínimo 15 caracteres): queda registrado en el archivo")
    v1 = Path(v1_test_evaluation)
    if not v1.is_file():
        raise TestAuthorizationError(f"no existe {v1}: la autorización se ata a la evaluación de test de v1 (primer uso)")
    out = Path(reports_dir) / FILENAME
    if out.exists():
        raise TestAuthorizationError(f"{out} ya existe: la autorización no se reescribe")
    doc = {
        "schema_version": SCHEMA,
        "authorized_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "authorized_by": "integrador del proyecto (Acosta)",
        "reason": reason.strip(),
        "split_use_number": 2,
        "first_use": {"model": "v1", "file": str(v1).replace("\\", "/"), "sha256": _sha(v1)},
        "declaration": (
            "El split test se abrió antes con v1 (primer uso). Al diseñar v2 se conocían las cifras de ese resultado, pero "
            "ninguna decisión de v2 (aumentación, rangos, escenarios de robustez, selección, umbral) se basó en predicciones "
            "ni en métricas de test: se tomaron con el corpus de entrenamiento, la validación y capturas de webcam. "
            "Este es un segundo uso declarado; deja de ser una evaluación única y así debe informarse."
        ),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out


def require_valid_authorization(reports_dir: str | Path, v1_test_evaluation: str | Path = V1_TEST_EVALUATION) -> dict:
    """Devuelve la autorización si es válida; si no, lanza ``TestAuthorizationError`` (el test sigue sellado)."""
    p = Path(reports_dir) / FILENAME
    if not p.is_file():
        raise TestAuthorizationError(
            "esta configuración tiene el test SELLADO (evaluation.test_sealed). Un segundo uso requiere una autorización "
            "explícita: `python -m src.lsc_models authorize-test --reason \"...\" --confirm-second-use`"
        )
    auth = json.loads(p.read_text(encoding="utf-8"))
    if auth.get("schema_version") != SCHEMA or auth.get("split_use_number") != 2:
        raise TestAuthorizationError(f"{p} no es una autorización válida de segundo uso")
    v1 = Path(v1_test_evaluation)
    if not v1.is_file() or _sha(v1) != auth.get("first_use", {}).get("sha256"):
        raise TestAuthorizationError("la autorización no corresponde al test_evaluation.json de v1 vigente (cambió o falta)")
    return auth
