"""Paridad pydantic <-> JSON Schema (design §5, §20).

Para cada contrato se construyen instancias válidas e inválidas; pydantic y el
JSON Schema correspondiente deben aceptar/rechazar EXACTAMENTE el mismo
conjunto.
"""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest
from pydantic import ValidationError

from contracts.models import FeatureSpec, LabelMap, ManifestRecord, SplitDoc

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMAS = REPO_ROOT / "contracts" / "schemas"
FIXTURES = REPO_ROOT / "contracts" / "fixtures"


def _schema(name: str) -> dict:
    return json.loads((SCHEMAS / name).read_text(encoding="utf-8"))


def _pydantic_accepts(model, payload: dict) -> bool:
    try:
        model.model_validate(payload)
        return True
    except ValidationError:
        return False


def _jsonschema_accepts(schema: dict, payload: dict) -> bool:
    try:
        jsonschema.validate(payload, schema)
        return True
    except jsonschema.ValidationError:
        return False


def _valid_manifest_record() -> dict:
    line = (FIXTURES / "SYNTHETIC_valid_sample.jsonl").read_text(encoding="utf-8").strip()
    return json.loads(line.splitlines()[0])


def _parity_cases():
    """(modelo, schema, payload, se_espera_valido) para cada contrato."""
    manifest_schema = _schema("manifest.schema.json")
    rec = _valid_manifest_record()

    bad_rec_5frames = json.loads(json.dumps(rec))
    bad_rec_5frames["frame_paths"] = rec["frame_paths"][:5]
    bad_rec_5frames["frame_sha256"] = rec["frame_sha256"][:5]
    bad_rec_5frames["frame_indices"] = rec["frame_indices"][:5]

    bad_rec_dupidx = json.loads(json.dumps(rec))
    bad_rec_dupidx["frame_indices"] = [0, 1, 2, 3, 4, 4]

    bad_rec_class = json.loads(json.dumps(rec))
    bad_rec_class["class_id"] = 42

    labels = json.loads((REPO_ROOT / "contracts" / "labels.json").read_text(encoding="utf-8"))
    bad_labels = json.loads(json.dumps(labels))
    bad_labels["labels"] = bad_labels["labels"][:9]  # sólo 9 clases

    splits = json.loads((FIXTURES / "SYNTHETIC_splits.json").read_text(encoding="utf-8"))
    # Inválido para AMBOS: una clave extra (pydantic extra="forbid",
    # jsonschema additionalProperties:false). El `seed` const sólo lo fija el
    # JSON Schema, así que no sirve como caso de paridad.
    bad_splits = json.loads(json.dumps(splits))
    bad_splits["campo_extra"] = "no permitido"

    spec = json.loads((REPO_ROOT / "contracts" / "feature_spec.json").read_text(encoding="utf-8"))
    bad_spec = json.loads(json.dumps(spec))
    bad_spec["point_order"] = spec["point_order"][:10]  # longitud != 49

    return [
        (ManifestRecord, manifest_schema, rec, True),
        (ManifestRecord, manifest_schema, bad_rec_5frames, False),
        (ManifestRecord, manifest_schema, bad_rec_dupidx, False),
        (ManifestRecord, manifest_schema, bad_rec_class, False),
        (LabelMap, _schema("labels.schema.json"), labels, True),
        (LabelMap, _schema("labels.schema.json"), bad_labels, False),
        (SplitDoc, _schema("split.schema.json"), splits, True),
        (SplitDoc, _schema("split.schema.json"), bad_splits, False),
        (FeatureSpec, _schema("feature_spec.schema.json"), spec, True),
        (FeatureSpec, _schema("feature_spec.schema.json"), bad_spec, False),
    ]


@pytest.mark.parametrize("model,schema,payload,expected", _parity_cases())
def test_pydantic_jsonschema_parity(model, schema, payload, expected):
    p = _pydantic_accepts(model, payload)
    j = _jsonschema_accepts(schema, payload)
    assert p == expected, f"pydantic {model.__name__}: esperado {expected}, obtuvo {p}"
    assert j == expected, f"jsonschema {schema.get('title')}: esperado {expected}, obtuvo {j}"
    assert p == j, "pydantic y jsonschema divergen en el mismo payload"
