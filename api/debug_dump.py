"""Volcado local de intentos reales para diagnóstico (DESACTIVADO por defecto).

Se activa con la variable de entorno ``LSC_DEBUG_DUMP_DIR`` (o ``run-app --debug-dump``). Por cada
intento que llega al trabajador guarda, en una subcarpeta propia:

- ``frame_0.jpg`` … ``frame_5.jpg``: los seis cuadros recibidos (**contienen el rostro de la persona**);
- ``x.npy``: la secuencia (6, 147) que vio el modelo;
- ``meta.json``: probabilidades, decisión, razones, presencia de manos y hombros por cuadro.

PRIVACIDAD: solo local; la carpeta está ignorada por Git (``.gitignore``) y NUNCA debe subirse ni
compartirse. No cambia la respuesta HTTP ni las predicciones, y jamás lanza excepciones.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

ENV_VAR = "LSC_DEBUG_DUMP_DIR"

# Orden de puntos del contrato (contracts/feature_spec.json): 7 de pose + 21 mano izq. + 21 mano der.
N_POINTS = 49
POSE_LEFT_SHOULDER, POSE_RIGHT_SHOULDER = 1, 2  # pose_11 y pose_12
LH_SLICE = slice(7, 28)
RH_SLICE = slice(28, 49)


def presence_per_frame(x: np.ndarray) -> dict:
    """Presencia por cuadro (canal de máscara m) de hombros y manos, a partir de ``x`` (6, 147)."""
    z = np.asarray(x, dtype=np.float32).reshape(-1, N_POINTS, 3)
    m = z[..., 2] > 0
    return {
        "left_shoulder": m[:, POSE_LEFT_SHOULDER].astype(int).tolist(),
        "right_shoulder": m[:, POSE_RIGHT_SHOULDER].astype(int).tolist(),
        "left_hand": m[:, LH_SLICE].any(axis=1).astype(int).tolist(),
        "right_hand": m[:, RH_SLICE].any(axis=1).astype(int).tolist(),
    }


def _safe(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "", text)[:12] or "x"


class DebugDump:
    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)

    @classmethod
    def from_env(cls) -> "DebugDump | None":
        raw = os.environ.get(ENV_VAR, "").strip()
        if not raw:
            return None
        d = Path(raw).expanduser()
        try:
            d.mkdir(parents=True, exist_ok=True)
        except OSError as exc:  # carpeta inválida: el servicio sigue sin volcado
            print(f"[debug-dump] no se pudo crear {d}: {exc}", file=sys.stderr)
            return None
        print(
            f"[debug-dump] ACTIVO -> {d.resolve()}  (guarda cuadros con el rostro; solo local, no subir a Git)",
            file=sys.stderr,
        )
        return cls(d)

    def save(self, *, raw_frames: list[bytes], meta: dict, result, labels: list[str]) -> Path | None:
        """Guarda un intento. Devuelve la carpeta o ``None`` si falló (nunca lanza)."""
        try:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            out = self.directory / f"{stamp}_{_safe(str(meta.get('client_request_id', '')))}"
            out.mkdir(parents=True, exist_ok=False)
            for i, data in enumerate(raw_frames):
                (out / f"frame_{i}.jpg").write_bytes(data)

            info = {
                "saved_at": time.time(),
                "width": meta.get("width"),
                "height": meta.get("height"),
                "observed_frame_count": meta.get("observed_frame_count"),
                "selected_indices": meta.get("selected_indices"),
                "timestamps_ms": meta.get("timestamps_ms"),
                "capture_duration_ms": meta.get("capture_duration_ms"),
                "quality_pass": bool(result.quality_pass),
                "accepted": bool(result.accepted),
                "reason_codes": list(result.reason_codes),
                "decision_label_id": result.label_id,
                "decision_confidence": result.confidence,
                "model_version": result.model_version,
                "timings_ms": dict(result.timings_ms),
            }
            x = getattr(result, "x", None)
            if x is not None:
                np.save(out / "x.npy", np.asarray(x, dtype=np.float32))
                info["presence"] = presence_per_frame(x)
            probs = getattr(result, "probs", None)
            if probs is not None:
                p = np.asarray(probs, dtype=np.float32).reshape(-1)
                order = np.argsort(p)[::-1][:3]
                info["probs"] = [float(v) for v in p]
                info["top3"] = [
                    {"label_id": int(i), "label": labels[int(i)] if int(i) < len(labels) else str(int(i)), "prob": float(p[int(i)])}
                    for i in order
                ]
            (out / "meta.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
            return out
        except Exception as exc:  # el diagnóstico nunca debe romper una petición
            print(f"[debug-dump] no se pudo guardar el intento: {exc}", file=sys.stderr)
            return None
