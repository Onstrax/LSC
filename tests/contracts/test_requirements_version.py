"""Compatibilidad básica de versiones y recolección (design §22, §20).

(a) ``deployment/requirements-data.in`` existe, es parseable y ancla versiones.
(b) ``pydantic`` es v2; Python 3.11 se verifica con skip/warning si el entorno
    es 3.13 (sin fabricar un rojo ajeno al código de Ortiz).
(c) ``tests/data``, ``tests/features`` y ``tests/contracts`` existen y son
    recolectables vía paths explícitos, SIN leer ``pytest.ini``.

NO compara reason_codes aquí (ver test_reason_codes_set.py).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pydantic
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
REQUIREMENTS = REPO_ROOT / "deployment" / "requirements-data.in"
TEST_DIRS = ("tests/data", "tests/features", "tests/contracts")


def _parse_pins(text: str) -> dict[str, str]:
    pins: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        # Reconoce anclaje exacto (==) o mínimo (>=) con versión.
        for sep in ("==", ">="):
            if sep in line:
                name, _, version = line.partition(sep)
                pins[name.strip().lower()] = version.strip()
                break
    return pins


# --- (a) requirements-data.in ----------------------------------------------
def test_requirements_data_in_exists_and_pins():
    assert REQUIREMENTS.is_file(), f"falta {REQUIREMENTS}"
    pins = _parse_pins(REQUIREMENTS.read_text(encoding="utf-8"))
    # Dependencias esperadas (design §22) ancladas a una versión.
    for pkg in ("numpy", "opencv-python-headless", "mediapipe", "pydantic",
                "jsonschema", "pyyaml", "pytest"):
        assert pkg in pins, f"{pkg} no está anclado en requirements-data.in"
        assert pins[pkg], f"{pkg} sin versión anclada"


# --- (b) versiones ---------------------------------------------------------
def test_pydantic_is_v2():
    assert pydantic.VERSION.startswith("2."), f"pydantic debe ser v2; es {pydantic.VERSION}"


def test_python_3_11_expected():
    """El diseño ancla Python 3.11; si el entorno difiere, se avisa sin romper."""
    major, minor = sys.version_info[:2]
    if (major, minor) != (3, 11):
        pytest.skip(
            f"entorno Python {major}.{minor} != 3.11 anclado por el diseño; "
            "la fusión al lock raíz (Acosta) congelará 3.11 (no rompe el verde)."
        )
    assert (major, minor) == (3, 11)


# --- (c) recolección vía paths explícitos (sin pytest.ini) -----------------
def test_test_dirs_exist():
    for d in TEST_DIRS:
        assert (REPO_ROOT / d).is_dir(), f"falta el directorio {d}"


def test_dirs_collectable_with_explicit_paths():
    """pytest --collect-only con paths explícitos recolecta las tres suites.

    Se usa ``-p no:cacheprovider`` y ``-o addopts=`` para no depender de la
    config de ``pytest.ini`` (que fija testpaths=tests/app,tests/e2e).
    """
    cmd = [
        sys.executable, "-m", "pytest", "--collect-only", "-q",
        "-p", "no:cacheprovider", "-o", "addopts=",
        *TEST_DIRS,
    ]
    proc = subprocess.run(
        cmd, cwd=str(REPO_ROOT), capture_output=True, text=True
    )
    assert proc.returncode == 0, f"la recolección falló:\n{proc.stdout}\n{proc.stderr}"
    out = proc.stdout
    # Se recolectan pruebas de las tres suites.
    assert "tests/data" in out or "tests\\data" in out
    assert "tests/features" in out or "tests\\features" in out
    assert "tests/contracts" in out or "tests\\contracts" in out
