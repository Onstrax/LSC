#!/usr/bin/env python3
"""Exporta el esquema OpenAPI de la API a api/schemas/openapi.json.

Este es el contrato PUBLICADO (lsc.http.v1) que Ortiz incorpora en contracts/.
Incluye ejemplos sintéticos de los tres estados cargados desde api/fixtures/.

    python deployment/export_openapi.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from api.fixtures import FIXTURES_DIR  # noqa: E402
from api.main import app  # noqa: E402


def main() -> int:
    schema = app.openapi()

    # Adjuntar ejemplos sintéticos de los tres estados al endpoint de predicción.
    examples = {}
    for name in ("accepted", "retry", "error"):
        with (FIXTURES_DIR / f"response_{name}.json").open(encoding="utf-8") as fh:
            examples[name] = {"value": json.load(fh)}

    try:
        predict = schema["paths"]["/api/v1/predict"]["post"]
        responses = predict.setdefault("responses", {})
        for code in ("200", "400", "413", "415", "422", "429", "503", "504", "500"):
            responses.setdefault(code, {"description": "Sobre JSON uniforme (ver docs)."})
        responses["200"].setdefault("content", {}).setdefault(
            "application/json", {}
        )["examples"] = {
            "accepted": examples["accepted"],
            "retry": examples["retry"],
        }
        responses["422"].setdefault("content", {}).setdefault(
            "application/json", {}
        )["examples"] = {"error": examples["error"]}
    except KeyError:
        pass  # el endpoint podría no aparecer si cambia la ruta

    out = ROOT / "api" / "schemas" / "openapi.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        json.dump(schema, fh, ensure_ascii=False, indent=2, sort_keys=True)
    print(f"OpenAPI exportado a {out} ({out.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
