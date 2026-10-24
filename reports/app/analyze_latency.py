#!/usr/bin/env python3
"""Analiza latency_attempts.csv y produce latency_summary.json (SDD §16).

Reglas:
- Percentil por rango más próximo: ordenar y tomar posición ceil(p*N), con
  posiciones desde 1.
- Los timeouts se reportan como CENSURADOS con su plazo, no como 0 ms ni como
  observaciones completas.
- Reportar N, p50, p95, media, errores, timeouts y tasa de rechazos.
- Distinguir accepted de retry/error; medir al menos 100 intentos válidos tras
  5 calentamientos. Este script NO inventa datos: requiere un CSV real.

    python reports/app/analyze_latency.py reports/app/latency_attempts.csv

Columnas esperadas (unidades en el nombre): attempt_id, source, mode,
api_version, model_version, feature_spec_hash, capture_protocol_version,
integral_ms, select_encode_ms, http_wait_ms, visual_update_ms, server_total_ms,
decode_ms, extraction_ms, inference_ms, decision_ms, status, reason_codes,
is_warmup, timed_out.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path


def nearest_rank_percentile(sorted_values: list[float], p: float) -> float | None:
    """Percentil por rango más próximo con posiciones desde 1."""
    n = len(sorted_values)
    if n == 0:
        return None
    rank = math.ceil(p * n)
    rank = max(1, min(rank, n))
    return sorted_values[rank - 1]


def to_float(v: str) -> float | None:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except ValueError:
        return None


def is_true(v: str) -> bool:
    return str(v).strip().lower() in {"true", "1", "yes"}


def analyze(path: Path) -> dict:
    rows = []
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            rows.append(r)

    warmups = [r for r in rows if is_true(r.get("is_warmup", ""))]
    valid = [r for r in rows if not is_true(r.get("is_warmup", ""))]

    timeouts = [r for r in valid if is_true(r.get("timed_out", ""))]
    non_timeout = [r for r in valid if not is_true(r.get("timed_out", ""))]

    by_status: dict[str, int] = {}
    for r in valid:
        s = r.get("status", "") or "unknown"
        by_status[s] = by_status.get(s, 0) + 1

    # Latencia integral SÓLO de intentos completados (no timeouts).
    integral = sorted(
        v
        for v in (to_float(r.get("integral_ms", "")) for r in non_timeout)
        if v is not None
    )
    # p95 de accepted específicamente.
    accepted_integral = sorted(
        v
        for v in (
            to_float(r.get("integral_ms", ""))
            for r in non_timeout
            if r.get("status") == "accepted"
        )
        if v is not None
    )

    errors = by_status.get("error", 0)
    rejects = by_status.get("retry", 0)
    n_valid = len(valid)

    summary = {
        "source_file": str(path),
        "n_total_rows": len(rows),
        "n_warmups": len(warmups),
        "n_valid": n_valid,
        "n_completed_non_timeout": len(non_timeout),
        "n_timeouts_censored": len(timeouts),
        "timeout_deadline_ms_note": "los timeouts se reportan censurados, no como 0 ms",
        "by_status": by_status,
        "errors": errors,
        "retries": rejects,
        "reject_rate": (rejects / n_valid) if n_valid else None,
        "error_rate": (errors / n_valid) if n_valid else None,
        "integral_ms": {
            "n": len(integral),
            "p50": nearest_rank_percentile(integral, 0.50),
            "p95": nearest_rank_percentile(integral, 0.95),
            "mean": (sum(integral) / len(integral)) if integral else None,
            "min": integral[0] if integral else None,
            "max": integral[-1] if integral else None,
        },
        "accepted_integral_ms": {
            "n": len(accepted_integral),
            "p50": nearest_rank_percentile(accepted_integral, 0.50),
            "p95": nearest_rank_percentile(accepted_integral, 0.95),
        },
        "targets": {
            "p95_target_ms": 2000,
            "valid_without_tech_failure_target": 0.95,
        },
    }

    # Evaluación de metas (sin inventar; sólo si hay datos suficientes).
    if accepted_integral:
        p95 = summary["accepted_integral_ms"]["p95"]
        summary["meets_p95_target"] = p95 is not None and p95 < 2000
    else:
        summary["meets_p95_target"] = None

    tech_failures = errors + len(timeouts)
    if n_valid:
        summary["valid_without_tech_failure_rate"] = (n_valid - tech_failures) / n_valid
    else:
        summary["valid_without_tech_failure_rate"] = None

    if n_valid < 100:
        summary.setdefault("warnings", []).append(
            f"Menos de 100 intentos válidos (N={n_valid}); medición insuficiente."
        )
    return summary


def main(argv: list[str] | None = None) -> int:
    argv = argv or sys.argv[1:]
    if not argv:
        print("Uso: analyze_latency.py <latency_attempts.csv>", file=sys.stderr)
        return 2
    path = Path(argv[0])
    if not path.exists():
        print(f"No existe {path}", file=sys.stderr)
        return 2
    summary = analyze(path)
    out = path.parent / "latency_summary.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nResumen escrito en {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
