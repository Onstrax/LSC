"""Tipos pydantic v2 del contrato de datos (design §4).

Todos los modelos usan ``ConfigDict(extra="forbid", frozen=True)``. Los nombres
de campo son exactamente los de la especificación. ``FeatureResult`` NO vive
aquí: es el dataclass frozen de ``src/lsc_features/types.py`` (design §8.1).
"""

from __future__ import annotations

import posixpath
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

__all__ = [
    "LabelEntry",
    "LabelMap",
    "FeatureSpec",
    "ManifestRecord",
    "NegativeRecord",
    "ExclusionRecord",
    "SplitAssignment",
    "SplitDoc",
    "ManifestReport",
    "SplitReport",
    "FeatureBuildReport",
    "ExperienceLSC",
    "SampleType",
]

_FORBID_FROZEN = ConfigDict(extra="forbid", frozen=True)

NUM_CLASSES = 10
TEMPORAL_LENGTH = 6
FEATURES_PER_FRAME = 147


# ---------------------------------------------------------------------------
# §4.1 Etiquetas
# ---------------------------------------------------------------------------
class LabelEntry(BaseModel):
    """Una clase del vocabulario de 10 señas."""

    model_config = _FORBID_FROZEN

    class_id: int = Field(ge=0, le=9)
    code: str
    display_text: str
    raw_label: str


class LabelMap(BaseModel):
    """Mapa de etiquetas en ORDEN CANÓNICO FIJO (10 clases)."""

    model_config = _FORBID_FROZEN

    schema_version: str
    labels: list[LabelEntry]

    @field_validator("labels")
    @classmethod
    def _ten_consecutive(cls, labels: list[LabelEntry]) -> list[LabelEntry]:
        if len(labels) != NUM_CLASSES:
            raise ValueError(f"labels debe tener {NUM_CLASSES} entradas")
        for index, entry in enumerate(labels):
            if entry.class_id != index:
                raise ValueError(
                    "class_id debe ser consecutivo 0..9 en orden; "
                    f"posición {index} tiene class_id={entry.class_id}"
                )
        return labels


# ---------------------------------------------------------------------------
# §4.2 Especificación de características
# ---------------------------------------------------------------------------
class FeatureSpec(BaseModel):
    """Especificación de características (design §4.2).

    ``feature_spec_hash`` NO forma parte de la firma que se hashea (design §6);
    ``build_date`` es un campo separado que NO entra al hash semántico.
    """

    model_config = _FORBID_FROZEN

    schema_version: str
    feature_version: str
    point_order: list[str]
    coordinate_system: str
    temporal_length: int = TEMPORAL_LENGTH
    features_per_frame: int = FEATURES_PER_FRAME
    dtype: str = "float32"
    missing_policy: dict
    quality_rules: dict
    detector_versions: dict
    model_asset_hashes: dict
    normalization: dict
    build_date: Optional[str] = None
    feature_spec_hash: Optional[str] = None

    @field_validator("point_order")
    @classmethod
    def _point_order_len(cls, point_order: list[str]) -> list[str]:
        if len(point_order) != 49:
            raise ValueError("point_order debe tener 49 nombres")
        return point_order

    @field_validator("temporal_length")
    @classmethod
    def _temporal(cls, value: int) -> int:
        if value != TEMPORAL_LENGTH:
            raise ValueError(f"temporal_length debe ser {TEMPORAL_LENGTH}")
        return value

    @field_validator("features_per_frame")
    @classmethod
    def _features(cls, value: int) -> int:
        if value != FEATURES_PER_FRAME:
            raise ValueError(f"features_per_frame debe ser {FEATURES_PER_FRAME}")
        return value

    @field_validator("dtype")
    @classmethod
    def _dtype(cls, value: str) -> str:
        if value != "float32":
            raise ValueError("dtype debe ser 'float32'")
        return value


# ---------------------------------------------------------------------------
# §4.3 Manifiesto
# ---------------------------------------------------------------------------
class ExperienceLSC(str, Enum):
    fluent = "fluent"
    interpreter = "interpreter"
    instructed = "instructed"
    unknown = "unknown"


class SampleType(str, Enum):
    known = "known"
    rest = "rest"
    gesture = "gesture"
    out_of_vocab = "out_of_vocab"


def _reject_unsafe_paths(paths: list[str]) -> list[str]:
    """Rechaza rutas absolutas, componentes ``..`` y separadores de Windows.

    Las rutas del manifiesto son relativas a ``data/raw/`` (design §7); nunca
    contienen nombres personales ni escapan del árbol de datos.
    """
    for raw in paths:
        if not raw or raw.strip() != raw:
            raise ValueError(f"ruta de frame vacía o con espacios: {raw!r}")
        if posixpath.isabs(raw) or (len(raw) >= 2 and raw[1] == ":"):
            raise ValueError(f"ruta de frame absoluta no permitida: {raw!r}")
        normalized = raw.replace("\\", "/")
        if normalized.startswith("/"):
            raise ValueError(f"ruta de frame absoluta no permitida: {raw!r}")
        parts = normalized.split("/")
        if ".." in parts:
            raise ValueError(f"ruta de frame con '..' no permitida: {raw!r}")
    return paths


class ManifestRecord(BaseModel):
    """Una línea JSON por muestra (design §4.3)."""

    model_config = _FORBID_FROZEN

    schema_version: str
    sample_id: str
    source: str
    source_version: str
    signer_id: str
    class_id: Optional[int] = Field(default=None, ge=0, le=9)
    raw_label: str
    repetition_id: int
    frame_paths: list[str]
    frame_sha256: list[str]
    frame_indices: list[int]
    width: int
    height: int
    n_frames: int
    sequence_sha256: str
    structural_status: str

    # Campos opcionales para capturas propias (design §4.3).
    domain: Optional[str] = None
    participant_group: Optional[str] = None
    experience_lsc: Optional[ExperienceLSC] = None
    capture_session: Optional[str] = None
    capture_protocol_version: Optional[str] = None
    consent_ref: Optional[str] = None
    timestamps_ms: Optional[list[int]] = None
    sample_type: Optional[SampleType] = None

    @field_validator("frame_paths")
    @classmethod
    def _safe_paths(cls, paths: list[str]) -> list[str]:
        return _reject_unsafe_paths(paths)

    @model_validator(mode="after")
    def _check_sequence(self) -> "ManifestRecord":
        if not (len(self.frame_paths) == len(self.frame_sha256) == len(self.frame_indices) == 6):
            raise ValueError("frame_paths, frame_sha256 y frame_indices deben tener 6 elementos")
        if len(set(self.frame_indices)) != len(self.frame_indices):
            raise ValueError("frame_indices deben ser únicos")
        if self.n_frames != 6:
            raise ValueError("n_frames debe ser 6 para una muestra válida")
        return self


class NegativeRecord(ManifestRecord):
    """Negativo: ``class_id`` siempre ``null``; se exporta con ``y=-1``."""

    model_config = _FORBID_FROZEN

    @field_validator("class_id")
    @classmethod
    def _must_be_null(cls, value: Optional[int]) -> Optional[int]:
        if value is not None:
            raise ValueError("NegativeRecord.class_id debe ser null")
        return value


class ExclusionRecord(BaseModel):
    """Fila de ``exclusions.csv`` (design §4.3)."""

    model_config = _FORBID_FROZEN

    artifact_id: str
    stage: str
    reason_code: str
    detail: str


# ---------------------------------------------------------------------------
# §4.4 Splits
# ---------------------------------------------------------------------------
class SplitAssignment(BaseModel):
    """Asignación de ``signer_id`` por conjunto (design §4.4)."""

    model_config = _FORBID_FROZEN

    train: list[str]
    val: list[str]
    test: list[str]


class SplitSampleIds(BaseModel):
    model_config = _FORBID_FROZEN

    train: list[str]
    val: list[str]
    test: list[str]


class SplitDoc(BaseModel):
    """Artefacto ``splits.json`` (design §4.4)."""

    model_config = _FORBID_FROZEN

    schema_version: str
    seed: int
    manifest_sha256: str
    version: str
    signer_ids: SplitAssignment
    sample_ids: SplitSampleIds
    class_coverage: dict
    incidents: list[str]


# ---------------------------------------------------------------------------
# §4.5 Reportes
# ---------------------------------------------------------------------------
class ManifestReport(BaseModel):
    model_config = _FORBID_FROZEN

    n_records: int
    n_valid: int
    n_excluded: int
    by_class: dict
    by_signer: dict
    exclusions_path: str
    manifest_path: str
    warnings: list[str]


class SplitReport(BaseModel):
    model_config = _FORBID_FROZEN

    n_train: int
    n_val: int
    n_test: int
    n_signers: int
    class_coverage: dict
    incidents: list[str]
    splits_path: str


class FeatureBuildReport(BaseModel):
    model_config = _FORBID_FROZEN

    n_total: int
    n_quality_pass: int
    n_failed: int
    failed_sample_ids: list[str]
    feature_spec_hash: str
    bundle_path: str
    per_reason_counts: dict
