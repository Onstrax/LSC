"""El lock de Python 3.11 (deployment/requirements-lock-py311.txt) sigue siendo un lock completo y coherente con pyproject."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
LOCK = ROOT / "deployment" / "requirements-lock-py311.txt"


def _pins() -> dict[str, str]:
    pins = {}
    for raw in LOCK.read_text(encoding="utf-8").splitlines():
        line = raw.split(";")[0].strip()          # quita marcadores de entorno
        if not line or line.startswith("#"):
            continue
        assert "==" in line, f"línea no fijada con ==: {raw!r}"
        name, _, ver = line.partition("==")
        pins[re.sub(r"[-_.]+", "-", name.strip().lower())] = ver.strip()
    return pins


def test_every_line_is_pinned_exactly_and_nothing_local_leaks():
    # Solo líneas de paquetes (la cabecera de comentarios menciona `pip install -e` a propósito).
    body = "\n".join(l for l in LOCK.read_text(encoding="utf-8").splitlines() if l.strip() and not l.lstrip().startswith("#"))
    assert not re.search(r"^-e\s|@ file|/work", body, flags=re.M)               # nada de rutas locales del contenedor
    pins = _pins()
    assert len(pins) > 60 and all(re.fullmatch(r"[0-9][0-9A-Za-z.\-+!]*", v) for v in pins.values())


def test_the_lock_covers_every_direct_dependency_of_pyproject():
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    direct = list(data["dependencies"])
    for extra in ("features", "train", "app", "dev"):
        direct += data["optional-dependencies"][extra]
    pins = _pins()
    missing = []
    for spec in direct:
        name = re.split(r"[<>=!~\[ ]", spec, maxsplit=1)[0]
        if re.sub(r"[-_.]+", "-", name.lower()) not in pins:
            missing.append(name)
    assert not missing, f"el lock no cubre dependencias directas de pyproject: {missing}"


def test_windows_incompatible_package_is_marked():
    # uvloop no existe en Windows: sin marcador el lock rompería la instalación allí.
    lines = [l for l in LOCK.read_text(encoding="utf-8").splitlines() if l.startswith("uvloop")]
    assert lines and "sys_platform" in lines[0]
