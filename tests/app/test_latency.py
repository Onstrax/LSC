"""Pruebas del análisis de latencia (SDD §16).

Verifican la definición de percentil (rango más próximo, ceil(p*N), posiciones
desde 1), el tratamiento de timeouts como censurados y la distinción de estados.
"""

from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
_spec = importlib.util.spec_from_file_location(
    "analyze_latency", ROOT / "reports" / "app" / "analyze_latency.py"
)
analyze_latency = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(analyze_latency)


def test_percentile_nearest_rank():
    vals = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]  # N=10
    # ceil(0.95*10)=10 -> posición 10 -> 100
    assert analyze_latency.nearest_rank_percentile(vals, 0.95) == 100
    # ceil(0.50*10)=5 -> posición 5 -> 50
    assert analyze_latency.nearest_rank_percentile(vals, 0.50) == 50


def test_percentile_single():
    assert analyze_latency.nearest_rank_percentile([42.0], 0.95) == 42.0


def test_percentile_empty():
    assert analyze_latency.nearest_rank_percentile([], 0.95) is None


def _write_csv(path: Path, rows: list[dict]) -> None:
    cols = [
        "attempt_id", "source", "mode", "api_version", "model_version",
        "feature_spec_hash", "capture_protocol_version", "integral_ms",
        "select_encode_ms", "http_wait_ms", "visual_update_ms",
        "server_total_ms", "decode_ms", "extraction_ms", "inference_ms",
        "decision_ms", "status", "reason_codes", "is_warmup", "timed_out",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})


def test_analyze_censors_timeouts(tmp_path):
    rows = []
    # 5 calentamientos
    for i in range(5):
        rows.append({"attempt_id": i, "status": "accepted", "integral_ms": 100, "is_warmup": "true"})
    # 3 aceptados válidos
    for i in range(3):
        rows.append({"attempt_id": 100 + i, "status": "accepted", "integral_ms": 500 + i, "is_warmup": "false"})
    # 1 timeout: NO debe contar como 0 ms ni como observación completa
    rows.append({"attempt_id": 200, "status": "error", "integral_ms": "", "is_warmup": "false", "timed_out": "true"})
    csv_path = tmp_path / "latency_attempts.csv"
    _write_csv(csv_path, rows)

    summary = analyze_latency.analyze(csv_path)
    assert summary["n_warmups"] == 5
    assert summary["n_valid"] == 4
    assert summary["n_timeouts_censored"] == 1
    # El timeout no entra en el cálculo de percentiles integrales.
    assert summary["integral_ms"]["n"] == 3
    assert summary["accepted_integral_ms"]["n"] == 3
    # Advertencia por N < 100.
    assert any("100" in w for w in summary.get("warnings", []))
