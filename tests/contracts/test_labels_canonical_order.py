"""Orden canónico de etiquetas (design §4.1, §20, AGENTS.md §5).

El orden es FIJO (0 Hola .. 9 Licor), no inferido de ``os.listdir`` ni
alfabético. ``labels.json`` lo serializa y ``LabelMap`` lo valida consecutivo.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from contracts.models import LabelMap

REPO_ROOT = Path(__file__).resolve().parents[2]
LABELS = json.loads((REPO_ROOT / "contracts" / "labels.json").read_text(encoding="utf-8"))

CANONICAL_CODES = [
    "hola", "buenos", "dias", "tardes", "noches",
    "yo", "nombre", "anos", "gustar", "licor",
]
CANONICAL_DISPLAY = [
    "Hola", "Buenos", "Días", "Tardes", "Noches",
    "Yo", "Nombre", "Años", "Gustar", "Licor",
]


def test_labels_json_canonical_order():
    model = LabelMap.model_validate(LABELS)
    assert [e.class_id for e in model.labels] == list(range(10))
    assert [e.code for e in model.labels] == CANONICAL_CODES
    assert [e.display_text for e in model.labels] == CANONICAL_DISPLAY


def test_order_not_alphabetical():
    """El orden canónico NO coincide con el alfabético (prueba que es fijo)."""
    codes = [e["code"] for e in LABELS["labels"]]
    assert codes != sorted(codes)


def test_non_consecutive_class_ids_rejected():
    bad = json.loads(json.dumps(LABELS))
    # Intercambiar class_id de dos entradas rompe la consecutividad en orden.
    bad["labels"][0]["class_id"] = 5
    bad["labels"][5]["class_id"] = 0
    with pytest.raises(ValidationError):
        LabelMap.model_validate(bad)
