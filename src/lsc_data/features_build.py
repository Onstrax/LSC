"""Orquestación de ``features`` y semántica de ``--resume`` (design §19, DO07).

Esta es la construcción REAL del bundle de características sobre el corpus:

1. Se carga el ``feature_spec`` canónico desde ``contracts/feature_spec.json``
   (fuente del ``feature_spec_hash`` que exige ``validate``); la config de
   features (``features_v1.yaml``) aporta ``quality_rules`` al extractor.
2. Se construye un ``RealDetector`` sobre los ``.task`` reales de
   ``artifacts/models/`` (parametrizables). Si falta un ``.task`` en el camino
   real -> ``ConfigError`` accionable (exit 2), sin fallback a ``mock``
   (AGENTS.md regla 3).
3. Se recorre el manifiesto EN ORDEN de ``sample_id``; cada muestra se extrae
   con ``FeatureExtractor.extract_sequence`` sobre 6 cuadros RGB uint8 cargados
   por el ``frame_loader`` (resuelve ``frame_paths`` contra ``--raw-dir``). Un
   fallo de UNA muestra se registra y no detiene el proceso.
4. Se persiste el bundle con ``packaging.write_bundle`` incluyendo la columna
   ``split`` (mapeada desde ``splits.json``), ``feature_spec.json`` y
   ``exclusions.csv``.

El detector y el ``frame_loader`` son INYECTABLES (parámetros opcionales) para
pruebas sin corpus ni ``.task`` reales; por defecto se usa ``RealDetector`` y un
cargador de disco con ``cv2`` (import perezoso).

Semántica de ``--resume`` (§19): antes de reanudar se recomputan y comparan
EXACTAMENTE dos hashes: ``feature_spec_hash`` (§6.1) y ``manifest_sha256``
(§6.4). Si AMBOS coinciden se reanuda; si CUALQUIERA difiere se ABORTA con exit
3 y un mensaje accionable, sin sobrescribir el bundle.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping

from contracts.canonical import feature_spec_hash as _feature_spec_hash
from contracts.canonical import manifest_sha256 as _manifest_sha256

from .errors import ConfigError, IntegrityError
from .packaging import FEATURE_SPEC_JSON

__all__ = ["run_features", "check_resume_compatibility", "make_frame_loader"]

EXIT_OK = 0

_MANIFEST_SHA_META = "manifest_sha256"

# feature_spec canónico (fuente del feature_spec_hash que escribe el bundle).
_CONTRACTS_FEATURE_SPEC = (
    Path(__file__).resolve().parents[2] / "contracts" / "feature_spec.json"
)
# .task reales por defecto (camino real; AGENTS.md regla 3: sin fallback a mock).
_DEFAULT_MODELS_DIR = Path("artifacts/models")
_POSE_TASK_NAME = "pose_landmarker.task"
_HANDS_TASK_NAME = "hand_landmarker.task"

FrameLoader = Callable[[Mapping[str, Any]], "list[Any]"]


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


def _load_canonical_feature_spec() -> dict:
    """Carga el ``feature_spec`` canónico desde ``contracts/feature_spec.json``.

    Es la fuente del ``feature_spec_hash`` que escribe el bundle y que
    ``validate`` recomputa; no se deriva de ``features_v1.yaml`` (que no es el
    feature_spec completo).
    """
    if not _CONTRACTS_FEATURE_SPEC.is_file():
        raise ConfigError(
            f"feature_spec canónico no encontrado: {_CONTRACTS_FEATURE_SPEC}"
        )
    try:
        data = json.loads(_CONTRACTS_FEATURE_SPEC.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(
            f"feature_spec canónico mal formado: {exc}"
        ) from exc
    if not isinstance(data, dict):
        raise ConfigError("feature_spec canónico no es un mapping")
    return data


def _load_splits_doc(splits_path: str | Path) -> dict:
    path = Path(splits_path)
    if not path.is_file():
        raise ConfigError(f"splits.json no encontrado: {path}")
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"splits.json mal formado: {exc}") from exc
    if not isinstance(doc, dict) or "sample_ids" not in doc:
        raise ConfigError("splits.json sin la clave 'sample_ids'")
    return doc


def _sample_id_to_split(splits_doc: Mapping[str, Any]) -> dict[str, str]:
    """Mapea cada ``sample_id`` a su split desde ``splits.json`` (§10)."""
    mapping: dict[str, str] = {}
    for split_name, ids in splits_doc["sample_ids"].items():
        for sample_id in ids:
            mapping[str(sample_id)] = str(split_name)
    return mapping


def _read_exclusions(path: str | Path) -> list[tuple[str, str, str, str]]:
    """Lee ``exclusions.csv`` (si existe) para re-escribirlo en el bundle."""
    import csv

    src = Path(path)
    if not src.is_file():
        return []
    rows: list[tuple[str, str, str, str]] = []
    with src.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header is None:
            return []
        for row in reader:
            padded = (list(row) + ["", "", "", ""])[:4]
            rows.append((padded[0], padded[1], padded[2], padded[3]))
    return rows


def make_frame_loader(raw_dir: str | Path) -> FrameLoader:
    """Crea un ``frame_loader`` que resuelve ``frame_paths`` contra ``raw_dir``.

    Antepone ``raw_dir`` a cada ruta relativa de ``record['frame_paths']`` y las
    abre con ``cv2`` (import perezoso), convirtiendo BGR->RGB. Devuelve los 6
    arreglos RGB uint8 en orden temporal (sin espejo).
    """
    root = Path(raw_dir)

    def _loader(record: Mapping[str, Any]) -> list[Any]:
        import cv2  # import perezoso: nunca en el import del paquete

        frames = []
        for rel_path in record["frame_paths"]:
            full = root / rel_path
            bgr = cv2.imread(str(full), cv2.IMREAD_COLOR)
            if bgr is None:
                raise FileNotFoundError(f"No se pudo leer el cuadro {full}")
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        return frames

    return _loader


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


def _build_detector(models_dir: str | Path):
    """Construye un ``RealDetector`` sobre los ``.task`` de ``models_dir``.

    Camino real por defecto; sin fallback a ``mock`` (AGENTS.md regla 3). Si
    falta un ``.task`` -> ``ConfigError`` accionable (exit 2).
    """
    from src.lsc_features.detector import RealDetector

    root = Path(models_dir)
    pose_task = root / _POSE_TASK_NAME
    hands_task = root / _HANDS_TASK_NAME
    for key, path in (("pose", pose_task), ("hands", hands_task)):
        if not path.is_file():
            raise ConfigError(
                f"features: falta el artefacto '{key}' en {path}; el camino real "
                "requiere los .task reales en modo 'real'. No hay fallback a "
                "'mock' (AGENTS.md regla 3)."
            )
    return RealDetector(pose_task, hands_task), pose_task, hands_task


def run_features(
    *,
    config_path: str | Path,
    manifest_path: str | Path,
    splits_path: str | Path,
    out_dir: str | Path,
    resume: bool,
    raw_dir: str | Path | None = None,
    models_dir: str | Path = _DEFAULT_MODELS_DIR,
    detector: Any | None = None,
    frame_loader: FrameLoader | None = None,
) -> int:
    """Ejecuta el comando ``features`` (construcción REAL del bundle, DO07).

    Con ``--resume`` verifica la compatibilidad del bundle existente (§19).

    El ``detector`` y el ``frame_loader`` son INYECTABLES para pruebas sin
    corpus ni ``.task`` reales; por defecto se usa ``RealDetector`` sobre
    ``models_dir`` y un cargador de disco que resuelve ``frame_paths`` contra
    ``raw_dir`` (requerido en el camino real). Devuelve ``EXIT_OK`` en éxito;
    las condiciones de error se propagan como ``ConfigError`` (exit 2) o
    ``IntegrityError`` (exit 3).
    """
    import numpy as np

    from src.lsc_features.extractor import FeatureExtractor

    from .packaging import write_bundle

    if resume:
        check_resume_compatibility(
            config_path=config_path,
            manifest_path=manifest_path,
            out_dir=out_dir,
        )

    # (b) feature_spec canónico + quality_rules de la config de features.
    feature_spec = _load_canonical_feature_spec()
    config = _load_feature_spec_from_config(config_path)
    extractor_config = dict(feature_spec)
    if isinstance(config.get("quality_rules"), dict):
        extractor_config["quality_rules"] = config["quality_rules"]

    # (c) detector real (o inyectado) + rutas de .task.
    if detector is not None:
        pose_path = Path(models_dir) / _POSE_TASK_NAME
        hands_path = Path(models_dir) / _HANDS_TASK_NAME
    else:
        detector, pose_path, hands_path = _build_detector(models_dir)

    # (d) extractor con detector inyectado.
    extractor = FeatureExtractor(
        extractor_config,
        {"pose": str(pose_path), "hands": str(hands_path)},
        detector=detector,
    )

    # Cargador de cuadros: real (contra raw_dir) o inyectado.
    if frame_loader is None:
        if raw_dir is None:
            raise ConfigError(
                "features: --raw-dir es obligatorio en el camino real para "
                "resolver frame_paths contra la raíz del corpus."
            )
        frame_loader = make_frame_loader(raw_dir)

    records = _load_manifest_records(manifest_path)
    records.sort(key=lambda rec: str(rec["sample_id"]))
    splits_doc = _load_splits_doc(splits_path)
    split_by_sample = _sample_id_to_split(splits_doc)

    xs: list[Any] = []
    ys: list[int] = []
    sample_ids: list[str] = []
    signer_ids: list[str] = []
    splits_col: list[str] = []
    quality_flags: list[bool] = []
    failed_sample_ids: list[str] = []

    for record in records:
        sample_id = str(record["sample_id"])
        split = split_by_sample.get(sample_id)
        if split is None:
            # Muestra sin split asignado (p. ej. excluida): no entra al bundle.
            continue
        try:
            frames = frame_loader(record)
            result = extractor.extract_sequence(frames)
        except Exception:
            failed_sample_ids.append(sample_id)
            continue

        xs.append(np.asarray(result.x, dtype=np.float32))
        class_id = record.get("class_id")
        ys.append(int(class_id) if class_id is not None else -1)
        sample_ids.append(sample_id)
        signer_ids.append(str(record.get("signer_id", "")))
        splits_col.append(split)
        quality_flags.append(bool(result.quality_pass))

    if hasattr(detector, "close"):
        try:
            detector.close()
        except Exception:  # pragma: no cover - cierre best-effort
            pass

    if not xs:
        raise IntegrityError(
            "features: no se extrajo ninguna muestra válida; el bundle estaría "
            f"vacío (muestras fallidas={len(failed_sample_ids)})."
        )

    exclusions = _read_exclusions(Path(manifest_path).with_name("exclusions.csv"))

    write_bundle(
        out_dir,
        X=np.stack(xs).astype(np.float32),
        y=ys,
        sample_id=sample_ids,
        signer_id=signer_ids,
        split=splits_col,
        quality_pass=quality_flags,
        feature_spec=feature_spec,
        exclusions=exclusions,
    )
    return EXIT_OK
