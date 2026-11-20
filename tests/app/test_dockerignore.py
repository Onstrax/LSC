"""El .dockerignore no debe excluir código que el Dockerfile copia a la imagen (ni dejar pasar datos con rostros).

Regresión real (20 nov. 2026): el patrón `**/debug_dump*/` excluyó api/debug_dump.py, la imagen se construyó "bien" y el
contenedor entró en bucle con ModuleNotFoundError. Docker ignora la barra final de un patrón, así que `x*/` también
alcanza archivos. Esta prueba reproduce la semántica relevante de .dockerignore sin necesitar Docker.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
# Lo que deployment/Dockerfile copia a la imagen (COPY ...).
COPIED = ("pyproject.toml", "api", "app", "src", "contracts", "deployment")


def _regex(pattern: str) -> re.Pattern:
    p = pattern.strip().rstrip("/")  # Docker limpia la ruta: la barra final NO distingue carpeta de archivo
    out, i = "", 0
    while i < len(p):
        if p.startswith("**/", i):
            out += "(?:.*/)?"
            i += 3
        elif p.startswith("**", i):
            out += ".*"
            i += 2
        elif p[i] == "*":
            out += "[^/]*"
            i += 1
        elif p[i] == "?":
            out += "[^/]"
            i += 1
        else:
            out += re.escape(p[i])
            i += 1
    return re.compile(out + r"$")


def _rules() -> list[tuple[bool, re.Pattern]]:
    rules = []
    for line in (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        neg = line.startswith("!")
        rules.append((neg, _regex(line[1:] if neg else line)))
    return rules


def is_ignored(rel: str) -> bool:
    """Un archivo queda fuera si él o alguna carpeta que lo contiene coincide con un patrón (la última regla gana)."""
    parts = rel.split("/")
    ignored = False
    for neg, rx in _rules():
        if any(rx.match("/".join(parts[: k + 1])) for k in range(len(parts))):
            ignored = not neg
    return ignored


def test_matcher_follows_docker_semantics():
    assert is_ignored("api/debug_dump.py") is False                    # el código NO se excluye
    assert is_ignored("artifacts/debug_dump/x/frame_0.jpg") is True    # capturas con rostros sí
    assert is_ignored("artifacts_v2/model.keras") is True
    assert is_ignored("Speak in Colombian Sign Language A Dynamic LSC70 Database/a.jpg") is True
    assert is_ignored("some/where/debug_dump_v2/frame_0.jpg") is True
    assert is_ignored("some/where/debug_dump/frame_0.jpg") is True
    assert is_ignored("reports/models/x.csv") is True


def test_the_old_buggy_pattern_would_have_been_caught():
    rx = _regex("**/debug_dump*/")
    assert rx.match("api/debug_dump.py")  # lo que ocurrió: el patrón con barra final alcanza al archivo de código


@pytest.mark.parametrize("name", COPIED)
def test_nothing_the_dockerfile_copies_is_excluded(name):
    base = ROOT / name
    files = [base] if base.is_file() else [p for p in base.rglob("*") if p.is_file()]
    bad = []
    for p in files:
        rel = p.relative_to(ROOT).as_posix()
        if "__pycache__" in rel or rel.endswith(".pyc"):
            continue                                                    # su exclusión es intencional
        if is_ignored(rel):
            bad.append(rel)
    assert not bad, f".dockerignore excluye archivos que el Dockerfile copia: {bad[:10]}"


def test_the_dockerfile_still_copies_what_this_test_assumes():
    text = (ROOT / "deployment" / "Dockerfile").read_text(encoding="utf-8")
    for token in ("COPY pyproject.toml", "COPY api/", "COPY app/", "COPY src/", "COPY contracts/", "COPY deployment/"):
        assert token in text, f"el Dockerfile ya no tiene `{token}`: actualice COPIED en esta prueba"
