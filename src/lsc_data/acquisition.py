"""Adquisición sin red (design §7).

``acquire --config configs/data/source.yaml --archive OPCIONAL``:

- Sin ``--archive``: NO descarga. Registra el bloque DO02 (URL/DOI/versión) y
  termina en 0 si sólo verifica, o lanza ``ConfigError`` (exit 2) si el config
  es inválido. Nunca simula una descarga.
- Con ``--archive``: valida procedencia contra ``source.yaml`` (SHA-256, tamaño
  esperados) y extrae de forma segura (``security.safe_extract``). Rutas
  peligrosas -> ``IntegrityError`` (exit 3). ``source.yaml`` inválido o sin
  procedencia congelada -> ``ConfigError`` (exit 2).

``source.json`` registra URL, DOI, versión, fecha UTC, tamaño y SHA-256.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .errors import ConfigError, IntegrityError

__all__ = [
    "SourceConfig",
    "load_source_config",
    "acquire",
]

DOI_DO02 = "10.17632/9ssyn8tff5.2"


@dataclass(frozen=True)
class SourceConfig:
    """Procedencia declarada en ``source.yaml`` (design §7)."""

    name: str
    doi: str
    url: str
    version: str
    subset: str
    expected_sha256: str | None
    expected_size_bytes: int | None
    allowed_extensions: tuple[str, ...]


def _load_yaml(path: str | Path) -> dict[str, Any]:
    try:
        import yaml  # import perezoso
    except ModuleNotFoundError as exc:  # pragma: no cover - depende del entorno
        raise ConfigError(
            "PyYAML no está instalado; instala deployment/requirements-data.in "
            "para leer configs/data/source.yaml."
        ) from exc

    config_path = Path(path)
    if not config_path.is_file():
        raise ConfigError(f"source config no encontrado: {config_path}")
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"source.yaml mal formado: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise ConfigError(f"source.yaml {config_path} no es un mapping")
    return dict(raw)


def load_source_config(path: str | Path) -> SourceConfig:
    """Resuelve ``source.yaml`` a ``SourceConfig`` validado (``ConfigError``)."""
    data = _load_yaml(path)

    def _require_str(key: str) -> str:
        value = data.get(key)
        if not isinstance(value, str) or not value:
            raise ConfigError(f"source.yaml: '{key}' ausente o no es string")
        return value

    expected_sha256 = data.get("expected_sha256")
    if expected_sha256 is not None and not isinstance(expected_sha256, str):
        raise ConfigError("source.yaml: 'expected_sha256' debe ser string o null")
    expected_size = data.get("expected_size_bytes")
    if expected_size is not None and (
        not isinstance(expected_size, int) or isinstance(expected_size, bool)
    ):
        raise ConfigError("source.yaml: 'expected_size_bytes' debe ser int o null")

    allowed = data.get("allowed_extensions", [])
    if not isinstance(allowed, (list, tuple)):
        raise ConfigError("source.yaml: 'allowed_extensions' debe ser lista")

    return SourceConfig(
        name=_require_str("name"),
        doi=_require_str("doi"),
        url=_require_str("url"),
        version=str(data.get("version", "")),
        subset=str(data.get("subset", data.get("name", ""))),
        expected_sha256=expected_sha256,
        expected_size_bytes=expected_size,
        allowed_extensions=tuple(str(e) for e in allowed),
    )


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _write_source_json(
    config: SourceConfig,
    *,
    dest_dir: Path,
    archive_sha256: str,
    archive_size: int,
) -> Path:
    source_json = {
        "name": config.name,
        "doi": config.doi,
        "url": config.url,
        "version": config.version,
        "subset": config.subset,
        "acquired_utc": datetime.now(timezone.utc).isoformat(),
        "size_bytes": archive_size,
        "sha256": archive_sha256,
    }
    out = dest_dir / "source.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(source_json, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return out


def acquire(
    config_path: str | Path,
    *,
    archive: str | Path | None = None,
    dest_dir: str | Path | None = None,
) -> list[str]:
    """Verifica procedencia o extrae de forma segura (design §7).

    Sin ``archive``: registra el bloque DO02 y devuelve ``[]`` (sólo verifica
    que el config es válido; nunca descarga). Con ``archive``: valida SHA-256 y
    tamaño contra ``source.yaml`` y extrae; devuelve la lista de rutas movidas.
    """
    import logging

    logger = logging.getLogger("lsc_data.acquire")
    config = load_source_config(config_path)

    if archive is None:
        logger.info(
            "DO02 bloqueado: no se descarga por red. Para adquirir el corpus, "
            "coloca el archivo local y ejecuta 'acquire --archive <ruta>'. "
            "DOI=%s url=%s version=%s",
            config.doi,
            config.url,
            config.version,
        )
        return []

    archive_path = Path(archive)
    if not archive_path.is_file():
        raise ConfigError(f"--archive no es un archivo: {archive_path}")

    if config.expected_sha256 is None or config.expected_size_bytes is None:
        raise ConfigError(
            "source.yaml no tiene procedencia congelada (expected_sha256 / "
            "expected_size_bytes son null); no se valida ni extrae (sin fallback)."
        )

    actual_sha256, actual_size = _sha256_file(archive_path)
    if actual_size != config.expected_size_bytes:
        raise IntegrityError(
            f"tamaño del archivo {actual_size} != esperado "
            f"{config.expected_size_bytes}"
        )
    if actual_sha256 != config.expected_sha256:
        raise IntegrityError(
            f"SHA-256 del archivo {actual_sha256} != esperado "
            f"{config.expected_sha256}"
        )

    from .security import safe_extract

    destination = Path(dest_dir) if dest_dir is not None else Path("data/raw") / config.subset
    moved = safe_extract(
        archive_path,
        destination,
        allowed_extensions=config.allowed_extensions or None,
    )
    _write_source_json(
        config,
        dest_dir=destination,
        archive_sha256=actual_sha256,
        archive_size=actual_size,
    )
    return moved
