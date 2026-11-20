"""Textos de la interfaz (app/js/messages.js), ejecutados con Node.

Se omite si Node no está instalado. Las pruebas de navegador (e2e) usan Playwright.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
MESSAGES = ROOT / "app" / "js" / "messages.js"
NODE = shutil.which("node")

pytestmark = pytest.mark.skipif(NODE is None, reason="Node no está instalado")


def _run(expr: str):
    script = f"import * as m from {json.dumps(MESSAGES.as_uri())}; console.log(JSON.stringify({expr}));"
    out = subprocess.run(
        [NODE, "--input-type=module", "-e", script], capture_output=True, text=True, timeout=30, check=True
    )
    return json.loads(out.stdout)


@pytest.mark.parametrize(
    "code",
    [
        "INSUFFICIENT_HANDS",
        "MULTIPLE_PEOPLE",
        "INSUFFICIENT_POSE",
        "HAND_ASSOCIATION_AMBIGUOUS",
        "GEOMETRY_OUTLIER",
        "INVALID_IMAGE",
        "SIZE_MISMATCH",
        "INVALID_FRAME_COUNT",
        "LOW_CONFIDENCE",
    ],
)
def test_each_quality_code_has_its_own_message(code):
    generic = _run("m.retryMessage([])")
    specific = _run(f"m.retryMessage([{json.dumps(code)}])")
    assert specific != generic and specific.strip()


def test_unknown_or_missing_codes_fall_back_to_generic():
    generic = _run("m.retryMessage([])")
    assert _run("m.retryMessage(null)") == generic
    assert _run('m.retryMessage(["OTRO"])') == generic


def test_hands_message_does_not_demand_both_hands():
    # La calidad exige >= 1 mano en >= 4 de 6 cuadros, no ambas manos.
    assert "ambas" not in _run('m.retryMessage(["INSUFFICIENT_HANDS"])').lower()


def test_error_messages():
    assert "ocupado" in _run('m.errorMessage(429, ["SERVICE_BUSY"])')
    assert "operador" in _run('m.errorMessage(503, ["MODEL_UNAVAILABLE"])')
    assert "tardó" in _run('m.errorMessage(504, ["INFERENCE_TIMEOUT"])')
    assert "técnico" in _run("m.errorMessage(500, [])")


def test_confidence_is_shown_as_model_score_not_as_certainty():
    note = _run("m.confidenceNote(0.8731)")
    assert "0,87" in note and "No es la probabilidad" in note
    for bad in ("null", "undefined", "1.5", "-0.1", '"x"'):
        assert _run(f"m.confidenceNote({bad})") == ""


def test_suggestion_note_asks_confirmation_by_default():
    assert "10 señas" in _run("m.suggestionNote(10)")
    assert "Confírmela" in _run("m.suggestionNote(10)")
    assert "Confírmela" in _run("m.suggestionNote(10, true)")
    assert "Confírmela" not in _run("m.suggestionNote(10, false)")
    assert "12 señas" in _run("m.suggestionNote(12)")  # no depende de un 10 fijo
