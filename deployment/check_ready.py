#!/usr/bin/env python3
"""Verifica que el servicio esté listo en modo REAL con versiones y hashes.

Consulta GET /api/v1/health y comprueba:
- HTTP 200 y ready=true
- mode == real (salvo que se permita mock explícitamente con --allow-mock)
- opcionalmente, model_version y feature_spec_hash esperados

Código de salida distinto de cero si falta alguna condición (apto para CI y para
`docker compose ... ps` seguido de verificación).

    python deployment/check_ready.py --url http://127.0.0.1:8000
    python deployment/check_ready.py --expect-model v1.2.3 --expect-hash abc123
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request


def fetch_health(url: str, timeout: float) -> tuple[int, dict | None]:
    req = urllib.request.Request(
        url.rstrip("/") + "/api/v1/health",
        headers={"Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            body = json.loads(resp.read().decode("utf-8"))
            return resp.status, body
    except urllib.error.HTTPError as exc:
        try:
            body = json.loads(exc.read().decode("utf-8"))
        except Exception:
            body = None
        return exc.code, body
    except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
        print(f"ERROR: no se pudo contactar el servicio: {exc}", file=sys.stderr)
        return 0, None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verifica readiness del servicio LSC.")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--allow-mock", action="store_true", help="aceptar mode=mock")
    parser.add_argument("--expect-model", default=None, help="model_version esperado")
    parser.add_argument("--expect-hash", default=None, help="feature_spec_hash esperado")
    args = parser.parse_args(argv)

    status, body = fetch_health(args.url, args.timeout)
    if status != 200 or not body:
        print(f"NO LISTO: HTTP {status}; cuerpo={body}", file=sys.stderr)
        return 2

    problems: list[str] = []
    if not body.get("ready"):
        problems.append("ready=false")
    mode = body.get("mode")
    if mode != "real" and not args.allow_mock:
        problems.append(f"mode={mode} (se esperaba real; use --allow-mock para permitir mock)")
    if args.expect_model and body.get("model_version") != args.expect_model:
        problems.append(
            f"model_version={body.get('model_version')} != esperado {args.expect_model}"
        )
    if args.expect_hash and body.get("feature_spec_hash") != args.expect_hash:
        problems.append(
            f"feature_spec_hash={body.get('feature_spec_hash')} != esperado {args.expect_hash}"
        )

    if problems:
        print("NO LISTO:\n  - " + "\n  - ".join(problems), file=sys.stderr)
        return 1

    print(
        "LISTO: "
        f"mode={mode} model_version={body.get('model_version')} "
        f"feature_spec_hash={body.get('feature_spec_hash')} "
        f"capture_protocol_version={body.get('capture_protocol_version')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
