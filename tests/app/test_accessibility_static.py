"""Comprobaciones estáticas de accesibilidad sobre index.html (SDD §13).

Son revisiones automáticas que COMPLEMENTAN, no sustituyen, el recorrido manual
con teclado, zoom al 200 % y ventana estrecha, ni la revisión con tecnologías de
asistencia. La conformidad WCAG completa requiere pruebas manuales y revisión
experta.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
HTML = (ROOT / "app" / "index.html").read_text(encoding="utf-8")
CSS = (ROOT / "app" / "styles.css").read_text(encoding="utf-8")


def test_has_lang_attribute():
    assert re.search(r"<html[^>]*lang=", HTML)


def test_has_viewport_meta():
    assert 'name="viewport"' in HTML


def test_single_h1():
    assert HTML.count("<h1") == 1


def test_uses_real_buttons():
    # Botones reales, no divs clicables.
    assert "<button" in HTML
    assert 'role="button"' not in HTML  # no simular botones


def test_aria_live_region_present():
    assert 'aria-live' in HTML


def test_select_has_label():
    # El selector de corrección tiene <label for=...>.
    assert re.search(r'<label[^>]*for="correctSelect"', HTML)


def test_skip_link_present():
    assert 'class="skip-link"' in HTML


def test_focus_visible_styles():
    # El CSS define foco visible.
    assert ":focus-visible" in CSS


def test_touch_target_min_height():
    # Objetivos táctiles amplios: min-height 44px en botones.
    assert "min-height: 44px" in CSS


def test_status_region_role():
    assert 'role="status"' in HTML
