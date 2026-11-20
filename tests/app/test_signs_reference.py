"""Referencias visuales de las señas (app/js/signs.js y app/assets/signs/manifest.json)."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
SIGNS_DIR = ROOT / "app" / "assets" / "signs"
MANIFEST = SIGNS_DIR / "manifest.json"
SIGNS_JS = ROOT / "app" / "js" / "signs.js"
NODE = shutil.which("node")
SAFE_FILE = re.compile(r"^[A-Za-z0-9_-]+\.(png|jpe?g|webp|gif|svg|mp4|webm)$", re.I)

needs_node = pytest.mark.skipif(NODE is None, reason="Node no está instalado")


def _labels() -> list[str]:
    doc = json.loads((ROOT / "contracts" / "labels.json").read_text(encoding="utf-8"))
    return [e["display_text"] for e in sorted(doc["labels"], key=lambda e: e["class_id"])]


def _signs() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))["signs"]


# --- manifiesto ---------------------------------------------------------------
def test_manifest_has_exactly_the_vocabulary():
    assert set(_signs()) == set(_labels())


def test_manifest_entries_are_well_formed_and_files_exist():
    for label, entry in _signs().items():
        assert set(entry) <= {"file", "alt", "tip", "credit"}, label
        f = entry.get("file")
        if f is None:
            continue
        assert SAFE_FILE.match(f), f"{label}: nombre de archivo no permitido: {f!r}"
        assert (SIGNS_DIR / f).is_file(), f"{label}: falta el archivo {f}"
        assert entry.get("alt"), f"{label}: falta 'alt' (texto para lectores de pantalla)"


def test_no_orphan_media_without_manifest_entry():
    referenced = {e["file"] for e in _signs().values() if e.get("file")}
    media = {p.name for p in SIGNS_DIR.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".gif", ".svg", ".mp4", ".webm"}}
    assert media <= referenced, f"archivos sin referenciar en el manifiesto: {sorted(media - referenced)}"


# --- lógica JS ----------------------------------------------------------------
def _run(expr: str):
    script = f"import * as m from {json.dumps(SIGNS_JS.as_uri())}; console.log(JSON.stringify({expr}));"
    out = subprocess.run([NODE, "--input-type=module", "-e", script], capture_output=True, text=True, timeout=30, check=True)
    return json.loads(out.stdout)


@needs_node
def test_reference_without_file_is_empty_not_invented():
    ref = _run('m.signReference({"Hola": {"file": null, "alt": null}}, "Hola")')
    assert ref["kind"] is None and ref["src"] is None
    assert _run('m.signReference({}, "Hola")')["kind"] is None
    assert _run('m.signReference(null, "Hola")')["kind"] is None


@needs_node
def test_reference_image_and_video():
    img = _run('m.signReference({"Hola": {"file": "hola.png", "alt": "Mano abierta", "credit": "Autor, CC BY"}}, "Hola")')
    assert img["kind"] == "image" and img["src"] == "/static/assets/signs/hola.png"
    assert img["alt"] == "Mano abierta" and img["credit"] == "Autor, CC BY"
    vid = _run('m.signReference({"Licor": {"file": "licor.webm"}}, "Licor")')
    assert vid["kind"] == "video"
    assert "Licor" in vid["alt"]  # alt por defecto si falta


@needs_node
@pytest.mark.parametrize(
    "bad",
    ["../secret.png", "a/b.png", "..\\\\x.png", "http://evil.test/x.png", "//evil.test/x.png", "x.exe", "x.png?a=1", "", " ", "x"],
)
def test_unsafe_file_names_are_rejected(bad):
    ref = _run(f'm.signReference({{"Hola": {{"file": {json.dumps(bad)}}}}}, "Hola")')
    assert ref["kind"] is None and ref["src"] is None


@needs_node
def test_label_lookup_ignores_prototype_properties():
    assert _run('m.signReference({}, "constructor")')["kind"] is None
    assert _run('m.signReference({}, "__proto__")')["kind"] is None


@needs_node
def test_missing_text_names_the_sign():
    assert "«Gustar»" in _run('m.missingReferenceText("Gustar")')


@needs_node
def test_load_manifest_tolerates_failures():
    script = f"""
import * as m from {json.dumps(SIGNS_JS.as_uri())};
const ok = await m.loadSignManifest(async () => ({{ ok: true, json: async () => ({{ signs: {{ Hola: {{ file: null }} }} }}) }}));
const notFound = await m.loadSignManifest(async () => ({{ ok: false }}));
const boom = await m.loadSignManifest(async () => {{ throw new Error("red"); }});
const badShape = await m.loadSignManifest(async () => ({{ ok: true, json: async () => ({{ signs: 5 }}) }}));
console.log(JSON.stringify({{ ok, notFound, boom, badShape }}));
"""
    out = subprocess.run([NODE, "--input-type=module", "-e", script], capture_output=True, text=True, timeout=30, check=True)
    r = json.loads(out.stdout)
    assert r["ok"] == {"Hola": {"file": None}}
    assert r["notFound"] == {} and r["boom"] == {} and r["badShape"] == {}
