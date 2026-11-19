"""origin_guard: mismo origen permitido en cualquier puerto; otros orígenes y hosts, bloqueados."""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client_factory(monkeypatch):
    monkeypatch.setenv("LSC_MODE", "mock")
    monkeypatch.delenv("LSC_ALLOWED_ORIGINS", raising=False)
    monkeypatch.delenv("LSC_ALLOWED_HOSTS", raising=False)
    import api.main as main_module

    def make(base_url):
        return TestClient(main_module.app, base_url=base_url)

    return make


def test_same_origin_on_any_port_is_allowed(client_factory):
    with client_factory("http://localhost:8123") as c:
        r = c.get("/api/v1/live", headers={"Origin": "http://localhost:8123"})
        assert r.status_code == 200


def test_static_script_loads_with_origin_header_on_custom_port(client_factory):
    # Regresión: Chromium envía Origin al cargar main.js (type=module); daba 403 fuera del puerto 8000.
    with client_factory("http://127.0.0.1:8765") as c:
        r = c.get("/static/js/main.js", headers={"Origin": "http://127.0.0.1:8765"})
        assert r.status_code == 200


def test_cross_site_origin_is_blocked(client_factory):
    with client_factory("http://localhost:8123") as c:
        r = c.get("/api/v1/live", headers={"Origin": "https://evil.example"})
        assert r.status_code == 403


def test_other_local_port_origin_is_blocked(client_factory):
    with client_factory("http://localhost:8123") as c:
        assert c.get("/api/v1/live", headers={"Origin": "http://localhost:3000"}).status_code == 403


def test_origin_with_same_host_but_other_scheme_authority_is_blocked(client_factory):
    with client_factory("http://localhost:8123") as c:
        assert c.get("/api/v1/live", headers={"Origin": "http://localhost:8123.evil.example"}).status_code == 403


def test_unknown_host_is_blocked_even_if_origin_matches(client_factory):
    with client_factory("http://attacker.example:8123") as c:
        r = c.get("/api/v1/live", headers={"Origin": "http://attacker.example:8123"})
        assert r.status_code == 403 and "Host" in r.json()["detail"]


def test_no_origin_header_still_allowed_for_cli(client_factory):
    with client_factory("http://localhost:8123") as c:
        assert c.get("/api/v1/live").status_code == 200
