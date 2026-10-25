"""Orquestación de ``features`` y semántica de ``--resume`` (design §19).

La construcción real del bundle sobre el corpus requiere el extractor
(``src/lsc_features``, FEAT-003) y datos/``.task`` reales (bloques DO05/DO07);
este lote no la ejercita. Lo que SÍ pertenece a ``src/lsc_data`` y se implementa
aquí es la comprobación de compatibilidad de ``--resume``:

Antes de reanudar se recomputan y comparan EXACTAMENTE dos hashes:

- ``feature_spec_hash`` (§6.1) del ``feature_spec.json`` que resultaría de la
  config actual vs el guardado en el bundle existente, y
- ``manifest_sha256`` (§6.4) del manifiesto de entrada actual vs el registrado en
  el bundle.

Si AMBOS coinciden se reanuda; si CUALQUIERA difiere se ABORTA con exit 3 y un
mensaje accionable (esperado vs encontrado), sin sobrescribir el bundle.
"""

from __future__ import annotations

import json
from pathlib import Path

from contracts.canonical import feature_spec_hash as _feature_spec_hash
from contracts.canonical import manifest_sha256 as _manifest_sha256

from .errors import ConfigError, IntegrityError
from .packaging import FEATURE_SPEC_JSON

__all__ = ["run_features", "check_resume_compatibility"]

EXIT_OK = 0

_MANIFEST_SHA_META = "manifest_sha256"


def _load_feature_spec_from_config(config_path: str | Path) -> dict:
    """Carga el feature_spec que resultaría de la config actual.

    Para este lote, la config de features apunta al ``feature_spec`` del
    contrato; se acepta YAML o JSON. ``PyYAML`` se importa de forma perezosa.
    """
    path = Path(config_path)
    if not path.is_file():
        raise ConfigError(f"config de features no encontrado: {path}")
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml
        except ModuleNotFoundError as exc:  # pragma: no cover - depende del entorno
            raise ConfigError(
                "PyYAML no está instalado; necesario para leer la config de features."
            ) from exc
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise ConfigError(f"config de features {path} no es un mapping")
    return data


def _load_manifest_records(manifest_path: str | Path) -> list[dict]:
    path = Path(manifest_path)
    if not path.is_file():
        raise ConfigError(f"manifiesto no encontrado: {path}")
    records: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def check_resume_compatibility(
    *,
    config_path: str | Path,
    manifest_path: str | Path,
    out_dir: str | Path,
) -> None:
    """Compara hashes del bundle existente con los recomputados (§19).

    Lanza ``IntegrityError`` (exit 3) si ``feature_spec_hash`` o
    ``manifest_sha256`` difieren, sin tocar el bundle. ``ConfigError`` (exit 2)
    si faltan entradas para comparar.
    """
    out = Path(out_dir)
    spec_path = out / FEATURE_SPEC_JSON
    if not spec_path.is_file():
        raise ConfigError(
            f"--resume sin bundle previo: falta {spec_path}"
        )
    existing_spec = json.loads(spec_path.read_text(encoding="utf-8"))
    existing_spec_hash = existing_spec.get("feature_spec_hash")
    existing_manifest_hash = existing_spec.get(_MANIFEST_SHA_META)

    current_spec = _load_feature_spec_from_config(config_path)
    current_spec_hash = _feature_spec_hash(current_spec)
    current_manifest_hash = _manifest_sha256(_load_manifest_records(manifest_path))

    if existing_spec_hash != current_spec_hash:
        raise IntegrityError(
            "feature_spec_hash divergente en --resume: "
            f"esperado(bundle)={existing_spec_hash} "
            f"encontrado(config)={current_spec_hash}; no se sobrescribe."
        )
    if existing_manifest_hash != current_manifest_hash:
        raise IntegrityError(
            "manifest_sha256 divergente en --resume: "
            f"esperado(bundle)={existing_manifest_hash} "
            f"encontrado(actual)={current_manifest_hash}; no se sobrescribe."
        )


def run_features(
    *,
    config_path: str | Path,
    manifest_path: str | Path,
    splits_path: str | Path,
    out_dir: str | Path,
    resume: bool,
) -> int:
    """Ejecuta el comando ``features``.

    Con ``--resume`` verifica la compatibilidad del bundle existente (§19). La
    construcción del bundle sobre datos reales requiere el extractor (FEAT-003)
    y artefactos reales (DO05/DO07): en este lote ese camino está bloqueado y se
    reporta como ``ConfigError`` accionable (exit 2), sin fabricar datos ni
    descargar nada (AGENTS.md: sin fallback silencioso).
    """
    if resume:
        check_resume_compatibility(
            config_path=config_path,
            manifest_path=manifest_path,
            out_dir=out_dir,
        )

    raise ConfigError(
        "features: la construcción del bundle sobre el corpus requiere el "
        "extractor (src/lsc_features, FEAT-003) y artefactos reales (bloques "
        "DO05/DO07). Desbloqueo documentado en docs/data/blocks.md. No se "
        "fabrican características ni se descarga nada."
    )
