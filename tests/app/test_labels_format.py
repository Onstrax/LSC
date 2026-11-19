"""load_labels acepta los formatos de labels.json y rechaza órdenes inválidos (modo real)."""

import json

import pytest

from api.config import CANONICAL_LABEL_ORDER, ConfigError, load_labels, load_settings

ENTRIES = [{"class_id": i, "code": t.lower(), "display_text": t, "raw_label": t.upper()} for i, t in enumerate(CANONICAL_LABEL_ORDER)]


def _labels(tmp_path, monkeypatch, doc):
    (tmp_path / "labels.json").write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setenv("LSC_MODE", "real")
    monkeypatch.setenv("LSC_ARTIFACTS_DIR", str(tmp_path))
    return load_labels(load_settings())


@pytest.mark.parametrize("doc", [
    list(CANONICAL_LABEL_ORDER),
    {"labels": list(CANONICAL_LABEL_ORDER)},
    {str(i): t for i, t in enumerate(CANONICAL_LABEL_ORDER)},
    {"labels": ENTRIES},          # contrato de Ortiz
    ENTRIES,
    {"labels": list(CANONICAL_LABEL_ORDER), "entries": ENTRIES},  # formato de export-artifacts
])
def test_accepted_formats_return_texts(tmp_path, monkeypatch, doc):
    assert _labels(tmp_path, monkeypatch, doc) == list(CANONICAL_LABEL_ORDER)


def test_contract_objects_are_sorted_by_class_id(tmp_path, monkeypatch):
    assert _labels(tmp_path, monkeypatch, {"labels": list(reversed(ENTRIES))}) == list(CANONICAL_LABEL_ORDER)


def test_wrong_order_is_rejected(tmp_path, monkeypatch):
    swapped = list(CANONICAL_LABEL_ORDER)
    swapped[0], swapped[1] = swapped[1], swapped[0]
    with pytest.raises(ConfigError, match="orden canónico"):
        _labels(tmp_path, monkeypatch, swapped)


@pytest.mark.parametrize("doc,msg", [
    ([ENTRIES[0]] * 10, "class_id"),
    (list(CANONICAL_LABEL_ORDER)[:9], "se esperaban"),
    ([{"class_id": i} for i in range(10)], "display_text"),
    (list(CANONICAL_LABEL_ORDER)[:9] + [ENTRIES[9]], "mezcla"),
])
def test_malformed_files_rejected(tmp_path, monkeypatch, doc, msg):
    with pytest.raises(ConfigError, match=msg):
        _labels(tmp_path, monkeypatch, doc)
