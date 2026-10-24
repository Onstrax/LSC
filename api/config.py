"""Configuración de la aplicación y carga de etiquetas.

Lee variables de entorno, valida tipos y rangos al arrancar, y resuelve el modo
(real/mock) y la ruta de artefactos. El orden canónico de etiquetas proviene de
``labels.json`` (Acosta); en modo mock se usa una lista equivalente para que la
UI tenga textos.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from api.constants import (
    CAPTURE_MAX_FPS,
    CAPTURE_PREP_MS,
    CAPTURE_PROTOCOL_VERSION,
    CAPTURE_WINDOW_MS,
    MAX_BODY_BYTES,
    MAX_HEIGHT,
    MAX_JPEG_BYTES,
    MAX_METADATA_BYTES,
    MAX_WIDTH,
    MODE_MOCK,
    MODE_REAL,
    NUM_CLASSES,
    SERVER_JOB_TIMEOUT_MS,
    UI_WAIT_TIMEOUT_MS,
)

# Orden canónico esperado (para validar labels.json de Acosta).
CANONICAL_LABEL_ORDER = (
    "Hola",
    "Buenos",
    "Días",
    "Tardes",
    "Noches",
    "Yo",
    "Nombre",
    "Años",
    "Gustar",
    "Licor",
)


class ConfigError(RuntimeError):
    """Configuración inválida detectada al arrancar."""


def _env_int(name: str, default: int, *, lo: int, hi: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} debe ser entero, se recibió {raw!r}.") from exc
    if not (lo <= value <= hi):
        raise ConfigError(f"{name}={value} fuera de rango [{lo}, {hi}].")
    return value


@dataclass(frozen=True)
class Settings:
    """Parámetros efectivos de la aplicación."""

    mode: str
    artifacts_dir: Path
    host: str
    port: int
    server_job_timeout_ms: int
    ui_wait_timeout_ms: int
    allowed_origins: tuple[str, ...]
    allowed_hosts: tuple[str, ...]
    mock_threshold: float

    @property
    def is_mock(self) -> bool:
        return self.mode == MODE_MOCK


def load_settings() -> Settings:
    """Construye :class:`Settings` desde el entorno, validando tipos y rangos."""
    mode = os.environ.get("LSC_MODE", MODE_REAL).strip().lower()
    if mode not in {MODE_REAL, MODE_MOCK}:
        raise ConfigError(
            f"LSC_MODE debe ser '{MODE_REAL}' o '{MODE_MOCK}', no {mode!r}."
        )

    artifacts_dir = Path(os.environ.get("LSC_ARTIFACTS_DIR", "artifacts")).expanduser()
    port = _env_int("LSC_PORT", 8000, lo=1, hi=65535)
    server_timeout = _env_int(
        "LSC_SERVER_TIMEOUT_MS", SERVER_JOB_TIMEOUT_MS, lo=1000, hi=120_000
    )
    ui_timeout = _env_int("LSC_UI_TIMEOUT_MS", UI_WAIT_TIMEOUT_MS, lo=1000, hi=120_000)

    origins = os.environ.get(
        "LSC_ALLOWED_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000"
    )
    hosts = os.environ.get("LSC_ALLOWED_HOSTS", "localhost,127.0.0.1")
    # Umbral del predictor SIMULADO. El mock reparte probabilidad entre 10
    # clases, así que el máximo rara vez supera 0.5. Un valor de 0.2 hace que el
    # flujo muestre sugerencias la mayoría de las veces y también algunos
    # "repita", de modo que quien clone el repo vea ambos estados. No afecta al
    # modo real, que usa la política y el umbral de Acosta.
    threshold_raw = os.environ.get("LSC_MOCK_THRESHOLD", "0.2")
    try:
        threshold = float(threshold_raw)
    except ValueError as exc:
        raise ConfigError(
            f"LSC_MOCK_THRESHOLD debe ser float, no {threshold_raw!r}."
        ) from exc
    if not (0.0 <= threshold <= 1.0):
        raise ConfigError("LSC_MOCK_THRESHOLD debe estar en [0, 1].")

    return Settings(
        mode=mode,
        artifacts_dir=artifacts_dir,
        host=os.environ.get("LSC_HOST", "127.0.0.1"),
        port=port,
        server_job_timeout_ms=server_timeout,
        ui_wait_timeout_ms=ui_timeout,
        allowed_origins=tuple(o.strip() for o in origins.split(",") if o.strip()),
        allowed_hosts=tuple(h.strip() for h in hosts.split(",") if h.strip()),
        mock_threshold=threshold,
    )


def load_labels(settings: Settings) -> list[str]:
    """Devuelve las etiquetas en orden canónico.

    - Modo real: lee ``labels.json`` y verifica el orden canónico.
    - Modo mock: usa la lista equivalente.
    """
    if settings.is_mock:
        return list(CANONICAL_LABEL_ORDER)

    labels_path = settings.artifacts_dir / "labels.json"
    if not labels_path.exists():
        raise ConfigError(
            f"No se encontró labels.json en {labels_path}. Requerido en modo real."
        )
    with labels_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    # Aceptar dos formatos: lista ordenada, o {"labels": [...]} / {"0": "Hola", ...}.
    if isinstance(data, dict) and "labels" in data:
        data = data["labels"]
    if isinstance(data, dict):
        labels = [data[str(i)] for i in range(NUM_CLASSES)]
    elif isinstance(data, list):
        labels = list(data)
    else:
        raise ConfigError("Formato de labels.json no reconocido.")

    if len(labels) != NUM_CLASSES:
        raise ConfigError(
            f"labels.json tiene {len(labels)} etiquetas; se esperaban {NUM_CLASSES}."
        )
    return labels


def capture_protocol_dict() -> dict:
    """Parámetros del protocolo de captura publicados a la UI."""
    return {
        "capture_protocol_version": CAPTURE_PROTOCOL_VERSION,
        "prep_ms": CAPTURE_PREP_MS,
        "window_ms": CAPTURE_WINDOW_MS,
        "max_fps": CAPTURE_MAX_FPS,
        "frame_count": 6,
        "jpeg_quality": 0.9,
        "ideal_width": MAX_WIDTH,
        "ideal_height": MAX_HEIGHT,
        "mirrored": False,
    }


def limits_dict() -> dict:
    """Límites publicados a la UI."""
    return {
        "max_body_bytes": MAX_BODY_BYTES,
        "max_jpeg_bytes": MAX_JPEG_BYTES,
        "max_metadata_bytes": MAX_METADATA_BYTES,
        "max_width": MAX_WIDTH,
        "max_height": MAX_HEIGHT,
        "max_pixels": MAX_WIDTH * MAX_HEIGHT,
    }
