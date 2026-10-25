"""Hash canónico del feature_spec (design §6.1, §20).

Estable ante rename cosmético o cambio de ``build_date``; cambia ante
``point_order``.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

from contracts.canonical import feature_spec_hash

REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC = json.loads((REPO_ROOT / "contracts" / "feature_spec.json").read_text(encoding="utf-8"))


def test_hash_matches_stored_value():
    """El hash recomputado coincide con el almacenado en feature_spec.json."""
    assert feature_spec_hash(SPEC) == SPEC["feature_spec_hash"]


def test_hash_stable_under_build_date_change():
    spec = copy.deepcopy(SPEC)
    base = feature_spec_hash(spec)
    spec["build_date"] = "2026-11-27T00:00:00Z"
    assert feature_spec_hash(spec) == base


def test_hash_stable_under_existing_hash_field():
    """El propio feature_spec_hash no entra al cálculo (campo *_hash excluido)."""
    spec = copy.deepcopy(SPEC)
    base = feature_spec_hash(spec)
    spec["feature_spec_hash"] = "otro-valor-cosmetico"
    assert feature_spec_hash(spec) == base


def test_hash_changes_under_point_order():
    spec = copy.deepcopy(SPEC)
    base = feature_spec_hash(spec)
    spec["point_order"] = list(reversed(spec["point_order"]))
    assert feature_spec_hash(spec) != base
