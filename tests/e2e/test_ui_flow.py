"""Pruebas de interfaz y operación U01–U15 (SDD §19) con Playwright.

La cámara simulada de Playwright se identifica como tal. Estas pruebas
automatizan estados, muestreo y recorridos repetibles; NO certifican
reconocimiento ni usabilidad humana. La prueba física de lados, el permiso del
navegador real y el piloto se documentan por separado.

Ejecución:
    python -m playwright install chromium
    python -m pytest tests/e2e -q
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.e2e


# --- Carga y estructura básica ---------------------------------------------
def test_loads_and_shows_vocabulary(page):
    assert page.locator("#vocabList li").count() == 10
    # Orden canónico: primera = Hola, última = Licor.
    assert page.locator("#vocabList li").first.inner_text() == "Hola"
    assert page.locator("#vocabList li").last.inner_text() == "Licor"


def test_mock_banner_visible(page):
    # En modo mock el rótulo persistente debe mostrarse.
    banner = page.locator("#banner")
    assert banner.is_visible()
    assert "prueba" in banner.inner_text().lower()


# --- U01: permiso / cámara (con cámara simulada concedida) ------------------
def test_u01_activate_camera(page):
    page.click("#activateBtn")
    # Con cámara simulada, debería llegar a READY y habilitar capturar.
    page.wait_for_selector("#captureBtn:not([disabled])", timeout=5000)
    assert page.locator("#captureBtn").is_enabled()


# --- U05: vista previa espejada, bytes sin espejo ---------------------------
def test_u05_preview_mirrored_class(page):
    # La vista previa usa la clase .mirrored (espejo sólo visual).
    assert "mirrored" in (page.locator("#video").get_attribute("class") or "")


# --- U13: recargar borra historial y no activa cámara -----------------------
def test_u13_reload_clears(page):
    page.reload()
    assert page.locator("#historyList li").count() == 0
    # La cámara no se activa sin interacción: capturar sigue deshabilitado.
    assert page.locator("#captureBtn").is_disabled()


# --- Sampling en el navegador (U03): verificar selectIndices expuesto -------
def test_u03_sampling_formula_in_browser(page):
    # Importa el módulo de captura y evalúa la fórmula en el navegador real.
    result = page.evaluate(
        """async () => {
            const mod = await import('/static/js/capture.js');
            return { n6: mod.selectIndices(6), n30: mod.selectIndices(30) };
        }"""
    )
    assert result["n6"] == [0, 1, 2, 3, 4, 5]
    assert result["n30"] == [0, 6, 12, 17, 23, 29]


# --- U10: navegación por teclado y foco visible -----------------------------
def test_u10_keyboard_focus(page):
    page.keyboard.press("Tab")
    focused = page.evaluate("() => document.activeElement && document.activeElement.tagName")
    # Algún control recibe foco (enlace de salto o botón).
    assert focused in {"A", "BUTTON", "SELECT"}
