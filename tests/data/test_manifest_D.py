"""Pruebas de manifiesto D01..D07, D11 (design §20).

Reconstrucción de secuencias, exclusiones, estabilidad de ``sample_id`` /
``sequence_sha256``, copia cruzada (D06, fallo deliberado), rango de
``class_id`` (D07) y determinismo from-scratch (D11). Sin red ni corpus real.
"""

from __future__ import annotations

import json

import pytest

from contracts.canonical import sample_id as canon_sample_id
from contracts.canonical import sequence_sha256 as canon_sequence_sha256
from contracts.models import ManifestRecord, NegativeRecord
from src.lsc_data.manifest import build_manifest_from_frames
from src.lsc_data.sequence import build_record, group_frames

from .conftest import LABEL_MAP, SOURCE, SOURCE_VERSION, make_sequence_frames


def _build_single(frames):
    candidates = group_frames(
        frames,
        source=SOURCE,
        source_version=SOURCE_VERSION,
        raw_label_by_class={v: k for k, v in LABEL_MAP.items()},
    )
    assert len(candidates) == 1
    return build_record(candidates[0])


# --- D01 -------------------------------------------------------------------
def test_D01_valid_reconstruction_stable_ids():
    """6 índices únicos ordenados -> ManifestRecord válida con ids estables."""
    frames = make_sequence_frames(indices=[5, 4, 3, 2, 1, 0])
    outcome = _build_single(frames)
    assert outcome.is_valid
    rec = outcome.record
    assert isinstance(rec, ManifestRecord)
    assert rec.n_frames == 6
    assert rec.frame_indices == [0, 1, 2, 3, 4, 5]  # ordenado temporalmente

    # sample_id y sequence_sha256 coinciden con la regla canónica (§6.2/§6.3).
    ordered_shas = rec.frame_sha256
    assert rec.sequence_sha256 == canon_sequence_sha256(ordered_shas)
    assert rec.sample_id == canon_sample_id(
        source_version=SOURCE_VERSION,
        signer_id=rec.signer_id,
        class_id=rec.class_id,
        repetition_id=rec.repetition_id,
        frame_sha256=ordered_shas,
    )

    # Reconstruir otra vez da exactamente los mismos ids (estabilidad).
    again = _build_single(make_sequence_frames(indices=[0, 1, 2, 3, 4, 5]))
    assert again.record.sample_id == rec.sample_id
    assert again.record.sequence_sha256 == rec.sequence_sha256


# --- D02 -------------------------------------------------------------------
def test_D02_missing_frames_excluded():
    """5 índices -> excluido (secuencia incompleta); no se completa con otra."""
    frames = make_sequence_frames(indices=[0, 1, 2, 3, 4])
    outcome = _build_single(frames)
    assert not outcome.is_valid
    assert outcome.reason_code == "INVALID_FRAME_COUNT"


# --- D03 -------------------------------------------------------------------
def test_D03_duplicate_ambiguous_excluded():
    """Dos frames con el mismo frame_index -> exclusión (ambiguo)."""
    frames = make_sequence_frames(indices=[0, 1, 2, 3, 4, 4])
    outcome = _build_single(frames)
    assert not outcome.is_valid
    assert outcome.reason_code == "SIZE_MISMATCH"


# --- D04 -------------------------------------------------------------------
def test_D04_mixed_people_or_classes_separated():
    """Grupos distintos por persona/clase no se mezclan en una sola muestra."""
    frames = make_sequence_frames(signer_id="S000", class_label="hola", class_id=0)
    frames += make_sequence_frames(signer_id="S001", class_label="hola", class_id=0)
    frames += make_sequence_frames(signer_id="S000", class_label="yo", class_id=5)
    candidates = group_frames(
        frames,
        source=SOURCE,
        source_version=SOURCE_VERSION,
        raw_label_by_class={v: k for k, v in LABEL_MAP.items()},
    )
    # Tres grupos distintos; ninguno mezcla persona/clase.
    assert len(candidates) == 3
    keys = {(c.signer_id, c.class_id) for c in candidates}
    assert keys == {("S000", 0), ("S001", 0), ("S000", 5)}
    for c in candidates:
        assert all(f.parsed.signer_id == c.signer_id for f in c.frames)
        assert all(f.parsed.class_id == c.class_id for f in c.frames)


# --- D05 -------------------------------------------------------------------
def test_D05_static_sign_identical_frames_valid():
    """Un signo estático con imágenes idénticas (hashes iguales) es válido."""
    same = "0" * 63 + "7"
    frames = make_sequence_frames(indices=range(6), shas=[same] * 6)
    outcome = _build_single(frames)
    assert outcome.is_valid
    assert outcome.record.frame_sha256 == [same] * 6


# --- D06 (fallo deliberado) ------------------------------------------------
def test_D06_cross_sequence_copy_detected(tmp_path):
    """sequence_sha256 igual entre signer_id distintos -> copia cruzada excluida.

    Fallo deliberado: si el control NO detecta la copia, ambas muestras
    quedarían válidas y esta prueba fallaría.
    """
    shared = [f"{i:064x}" for i in range(1, 7)]
    frames = make_sequence_frames(signer_id="S000", shas=shared)
    frames += make_sequence_frames(signer_id="S001", shas=shared)

    manifest_path = tmp_path / "manifest.jsonl"
    exclusions_path = tmp_path / "exclusions.csv"
    report = build_manifest_from_frames(
        frames,
        source=SOURCE,
        source_version=SOURCE_VERSION,
        label_map=LABEL_MAP,
        manifest_path=manifest_path,
        exclusions_path=exclusions_path,
    )

    # El control debe DETECTAR la copia: 0 válidas, 2 excluidas como incidente.
    assert report.n_records == 2
    assert report.n_valid == 0, "copia cruzada no detectada (defecto no capturado)"
    assert report.n_excluded == 2
    text = exclusions_path.read_text(encoding="utf-8")
    assert "copia cruzada" in text


# --- D07 -------------------------------------------------------------------
def test_D07_class_id_out_of_range_rejected():
    """class_id fuera de 0..9 lo rechaza el modelo (loader no acepta y fuera)."""
    base = make_sequence_frames(indices=range(6))
    outcome = _build_single(base)
    rec = outcome.record.model_dump()
    rec["class_id"] = 42
    with pytest.raises(Exception):
        ManifestRecord(**rec)
    rec["class_id"] = -1
    with pytest.raises(Exception):
        ManifestRecord(**rec)


def test_D07_negatives_go_to_y_minus_1_package():
    """Los negativos van a paquete propio (class_id=null -> y=-1), no a 10 clases."""
    base = make_sequence_frames(indices=range(6))
    rec = base and _build_single(base).record.model_dump()
    rec["class_id"] = None
    neg = NegativeRecord(**rec)
    assert neg.class_id is None
    # class_id no-nulo en un negativo es inválido.
    rec["class_id"] = 0
    with pytest.raises(Exception):
        NegativeRecord(**rec)


# --- D11 -------------------------------------------------------------------
def test_D11_determinism_from_scratch(tmp_path):
    """Dos ejecuciones from-scratch -> mismos sample_id, etiquetas y manifiesto."""
    frames = make_sequence_frames(signer_id="S002", class_label="yo", class_id=5)

    def run(dirname):
        d = tmp_path / dirname
        d.mkdir()
        build_manifest_from_frames(
            frames,
            source=SOURCE,
            source_version=SOURCE_VERSION,
            label_map=LABEL_MAP,
            manifest_path=d / "manifest.jsonl",
            exclusions_path=d / "exclusions.csv",
        )
        return (d / "manifest.jsonl").read_text(encoding="utf-8")

    first = run("a")
    second = run("b")
    assert first == second
    rec = json.loads(first.strip().splitlines()[0])
    assert rec["class_id"] == 5
    assert rec["signer_id"] == "S002"
