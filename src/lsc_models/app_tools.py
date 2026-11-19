"""Lanzar y comprobar la aplicación completa (API + interfaz).

``run-app``  arranca uvicorn (mock o real) tras comprobar los prerrequisitos.
``check-app`` prueba un servidor en marcha por HTTP: liveness, health, config, página y un
intento completo (6 JPEG + metadatos) además de peticiones inválidas. Sin dependencias extra
(urllib) salvo Pillow para fabricar los JPEG de la prueba.
"""

from __future__ import annotations

import io
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

REAL_FILES = ["model.keras", "labels.json", "feature_spec.json", "preprocessing.json", "thresholds.json",
              "manifest.json", "pose_landmarker.task", "hand_landmarker.task"]


def real_prerequisites(artifacts: Path) -> list[str]:
    return [f for f in REAL_FILES if not (artifacts / f).is_file()]


def _env(mode: str, artifacts: Path, port: int) -> dict:
    env = dict(os.environ)
    env.update({"LSC_MODE": mode, "LSC_ARTIFACTS_DIR": str(artifacts), "LSC_PORT": str(port), "TF_CPP_MIN_LOG_LEVEL": "3"})
    return env


def run_app(mode: str, artifacts: str, port: int, open_browser: bool) -> int:
    art = Path(artifacts)
    if mode == "real":
        missing = real_prerequisites(art)
        if missing:
            print(f"Modo real: faltan en {art}: {', '.join(missing)}\n"
                  "  - modelo/política: `python -m src.lsc_models export-artifacts`\n"
                  "  - .task: `python -m src.lsc_models download-models` y luego export-artifacts\n"
                  "  Para ver la interfaz sin modelo: `run-app --mode mock`.", file=sys.stderr)
            return 3
    print(f"Arrancando en modo {mode.upper()} -> http://localhost:{port}  (Ctrl+C para detener)")
    if open_browser:
        import threading
        import webbrowser

        threading.Timer(3.0, lambda: webbrowser.open(f"http://localhost:{port}")).start()
    return subprocess.call([sys.executable, "-m", "uvicorn", "api.main:app", "--host", "127.0.0.1", "--port", str(port)],
                           env=_env(mode, art, port))


# ----------------------------------------------------------------------------
def _http(method: str, url: str, body: bytes | None = None, headers: dict | None = None, timeout: int = 60):
    req = urllib.request.Request(url, data=body, method=method, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def _jpeg(i: int, w: int = 320, h: int = 240) -> bytes:
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (w, h), (200 - i * 10, 200, 210))
    ImageDraw.Draw(img).ellipse((w // 2 - 30, h // 2 - 30 + i * 3, w // 2 + 30, h // 2 + 30 + i * 3), fill=(120, 90, 70))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=85)
    return buf.getvalue()


def _multipart(frames: list[bytes], metadata: str) -> tuple[bytes, str]:
    b = uuid.uuid4().hex
    out = bytearray()
    for i, f in enumerate(frames):
        out += f'--{b}\r\nContent-Disposition: form-data; name="frames"; filename="frame_{i}.jpg"\r\nContent-Type: image/jpeg\r\n\r\n'.encode() + f + b"\r\n"
    out += f'--{b}\r\nContent-Disposition: form-data; name="metadata"\r\n\r\n{metadata}\r\n--{b}--\r\n'.encode()
    return bytes(out), f"multipart/form-data; boundary={b}"


def _meta(protocol: str, **over) -> str:
    m = {"schema_version": "lsc.http.v1", "client_request_id": str(uuid.uuid4()), "capture_protocol_version": protocol,
         "source": "authorized_fixture", "width": 320, "height": 240, "frame_count": 6, "observed_frame_count": 6,
         "selected_indices": [0, 1, 2, 3, 4, 5], "timestamps_ms": [0, 200, 400, 600, 800, 1000],
         "capture_duration_ms": 1000, "mirrored": False}
    m.update(over)
    return json.dumps(m)


def check_app(url: str, expect_mode: str | None = None) -> int:
    url = url.rstrip("/")
    results: list[tuple[bool, str]] = []

    def ok(cond: bool, name: str, detail: str = "") -> bool:
        results.append((cond, name))
        print(f"[{' OK ' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
        return cond

    try:
        st, _ = _http("GET", f"{url}/api/v1/live", timeout=10)
    except Exception as exc:
        print(f"No se pudo conectar con {url}: {exc}", file=sys.stderr)
        return 2
    ok(st == 200, "GET /api/v1/live", f"HTTP {st}")

    st, body = _http("GET", f"{url}/api/v1/health")
    h = json.loads(body) if body else {}
    if not ok(st == 200 and h.get("ready") is True, "GET /api/v1/health listo", f"HTTP {st} {json.dumps(h, ensure_ascii=False)[:160]}"):
        print("   El servicio no está listo (ver operator-guide: faltan artefactos o hashes incompatibles).")
        return 1
    if expect_mode:
        ok(h.get("mode") == expect_mode, f"modo = {expect_mode}", f"modo real del servidor: {h.get('mode')}")

    st, body = _http("GET", f"{url}/api/v1/config")
    cfg = json.loads(body)
    labels = cfg.get("labels", [])
    ok(st == 200 and len(labels) == 10 and all(isinstance(x, str) for x in labels), "GET /api/v1/config: 10 etiquetas de texto", ", ".join(map(str, labels))[:80])
    proto = cfg["capture_protocol"]["capture_protocol_version"]

    st, body = _http("GET", f"{url}/")
    ok(st == 200 and b"<html" in body.lower(), "GET / sirve la interfaz web", f"HTTP {st}")

    frames = [_jpeg(i) for i in range(6)]
    data, ctype = _multipart(frames, _meta(proto))
    t0 = time.perf_counter()
    st, body = _http("POST", f"{url}/api/v1/predict", data, {"Content-Type": ctype})
    ms = (time.perf_counter() - t0) * 1000
    r = json.loads(body)
    keys = {"schema_version", "request_id", "status", "label_id", "display_text", "confidence", "reason_codes", "mode", "timings_ms"}
    ok(st == 200 and keys <= set(r) and r["status"] in ("accepted", "retry"), "POST /predict (6 JPEG válidos) -> sobre válido",
       f"HTTP {st}, status={r.get('status')}, razones={r.get('reason_codes')}, {ms:.0f} ms")
    if r.get("status") == "accepted":
        ok(isinstance(r["display_text"], str) and r["display_text"] in labels, "display_text es una etiqueta de texto", str(r["display_text"]))

    data, ctype = _multipart(frames[:5], _meta(proto))
    st, body = _http("POST", f"{url}/api/v1/predict", data, {"Content-Type": ctype})
    ok(st in (400, 422) and json.loads(body)["status"] == "error", "POST con 5 cuadros -> error de validación", f"HTTP {st}")

    data, ctype = _multipart(frames, "esto no es json")
    st, body = _http("POST", f"{url}/api/v1/predict", data, {"Content-Type": ctype})
    ok(st in (400, 422) and json.loads(body)["status"] == "error", "POST con metadata ilegible -> error", f"HTTP {st}")

    data, ctype = _multipart([b"no soy un jpeg"] * 6, _meta(proto))
    st, body = _http("POST", f"{url}/api/v1/predict", data, {"Content-Type": ctype})
    ok(st in (415, 422) and json.loads(body)["status"] == "error", "POST con archivos que no son JPEG -> error", f"HTTP {st}")

    n_fail = sum(1 for c, _ in results if not c)
    print(f"\nRESULTADO: {len(results) - n_fail}/{len(results)} comprobaciones OK")
    return 0 if n_fail == 0 else 1


def check_app_spawn(mode: str, artifacts: str) -> int:
    """Arranca un servidor temporal, lo comprueba y lo detiene."""
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "api.main:app", "--host", "127.0.0.1", "--port", str(port)],
                            env=_env(mode, Path(artifacts), port), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(120):
            try:
                if _http("GET", f"http://127.0.0.1:{port}/api/v1/live", timeout=2)[0] == 200:
                    break
            except Exception:
                time.sleep(0.5)
        else:
            print("El servidor no arrancó en 60 s", file=sys.stderr)
            return 2
        # en modo real el modelo carga tras 'live': esperar readiness
        for _ in range(120):
            st, body = _http("GET", f"http://127.0.0.1:{port}/api/v1/health")
            if st == 200:
                break
            time.sleep(0.5)
        return check_app(f"http://127.0.0.1:{port}", expect_mode=mode)
    finally:
        proc.terminate()
        try:
            proc.wait(10)
        except subprocess.TimeoutExpired:
            proc.kill()
