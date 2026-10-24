"""Fixtures de pruebas e2e con Playwright para Python (SDD §15, §19).

Arrancan un servidor uvicorn real en modo mock y abren un navegador Chromium
con una cámara simulada. La cámara simulada de Playwright se identifica como
tal: estas pruebas NO sustituyen la prueba física de lados ni el piloto humano.

Requisitos (sólo entorno de desarrollo, no la imagen de ejecución):
    pip install -r deployment/requirements-dev.txt
    python -m playwright install chromium
"""

from __future__ import annotations

import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent

# Si Playwright no está instalado (sólo dependencia de desarrollo), se omite la
# recolección de las pruebas e2e sin abortar el resto de la suite.
try:
    from playwright.sync_api import sync_playwright  # noqa: F401

    _PLAYWRIGHT_AVAILABLE = True
except Exception:  # pragma: no cover - entorno sin Playwright
    _PLAYWRIGHT_AVAILABLE = False
    collect_ignore_glob = ["test_*.py"]


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def server():
    """Arranca uvicorn en modo mock y espera readiness."""
    port = _free_port()
    env = {
        "LSC_MODE": "mock",
        "LSC_HOST": "127.0.0.1",
        "LSC_PORT": str(port),
        "PATH": __import__("os").environ.get("PATH", ""),
    }
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "api.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        cwd=str(ROOT),
        env=env,
    )
    base = f"http://127.0.0.1:{port}"
    # Esperar readiness.
    import urllib.request

    ready = False
    for _ in range(50):
        try:
            with urllib.request.urlopen(f"{base}/api/v1/health", timeout=1) as r:
                if r.status == 200:
                    ready = True
                    break
        except Exception:
            time.sleep(0.2)
    if not ready:
        proc.terminate()
        pytest.fail("El servidor mock no alcanzó readiness.")
    yield base
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


@pytest.fixture()
def browser_context():
    """Chromium con cámara simulada y permisos concedidos."""
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=[
                "--use-fake-ui-for-media-stream",
                "--use-fake-device-for-media-stream",
            ],
        )
        context = browser.new_context(permissions=["camera"])
        yield context
        context.close()
        browser.close()


@pytest.fixture()
def page(browser_context, server):
    page = browser_context.new_page()
    page.goto(server)
    yield page
    page.close()
