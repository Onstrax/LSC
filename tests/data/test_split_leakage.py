"""Pruebas de splits y fuga D08..D10 (design §10, §20).

- D08: 70 personas -> 49/7/14 exactos con seed=20261022.
- D09 (fallo deliberado): fuga de identidad (misma secuencia en dos personas en
  splits distintos) -> IntegrityError (exit 3). Si el control no la detecta, la
  prueba falla.
- D10: falta una clase en un split -> incidente con conteos, sin recomputar
  semilla.
"""

from __future__ import annotations

import json

import pytest

from contracts.canonical import sample_id as canon_sample_id
from contracts.canonical import sequence_sha256 as canon_sequence_sha256
from src.lsc_data.errors import IntegrityError
from src.lsc_data.split import make_splits, split_counts

SOURCE_VERSION = "SYNTHETIC-v0"


def _record(signer_id: str, class_id: int, shas: list[str]) -> dict:
    seq = canon_sequence_sha256(shas)
    sid = canon_sample_id(
        source_version=SOURCE_VERSION,
        signer_id=signer_id,
        class_id=class_id,
        repetition_id=0,
        frame_sha256=shas,
    )
    return {
        "schema_version": "lsc.manifest.v1",
        "sample_id": sid,
        "source": "SYNTHETIC",
        "source_version": SOURCE_VERSION,
        "signer_id": signer_id,
        "class_id": class_id,
        "raw_label": str(class_id),
        "repetition_id": 0,
        "frame_paths": [f"lsc70_v2/{signer_id}/{class_id}/0/frame_{i}.jpg" for i in range(6)],
        "frame_sha256": shas,
        "frame_indices": [0, 1, 2, 3, 4, 5],
        "width": 640,
        "height": 480,
        "n_frames": 6,
        "sequence_sha256": seq,
        "structural_status": "valid",
    }


def _unique_shas(tag: str) -> list[str]:
    return ["%064x" % (abs(hash(f"{tag}|{i}")) % (16 ** 64)) for i in range(6)]


def _write_manifest(path, records: list[dict]) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False))
            fh.write("\n")


# --- D08 -------------------------------------------------------------------
def test_D08_split_counts_70():
    """split_counts(70) -> 49/7/14 (half-up determinista, §10)."""
    assert split_counts(70) == (49, 7, 14)
    # Caso de desempate .5 (N=25): half-up da n_val=3 (floor(2.5+0.5)).
    assert split_counts(25) == (25 - 3 - 5, 3, 5)


def test_D08_make_splits_70_people_exact(tmp_path):
    """70 personas (cada una con las 10 clases) -> 49/7/14 con seed canónico."""
    records = []
    for s in range(70):
        signer = f"S{s:03d}"
        for c in range(10):
            records.append(_record(signer, c, _unique_shas(f"{signer}-{c}")))
    manifest = tmp_path / "manifest.jsonl"
    _write_manifest(manifest, records)

    report = make_splits(manifest, out_path=tmp_path / "splits.json")
    assert (report.n_train, report.n_val, report.n_test) == (49, 7, 14)
    assert report.n_signers == 70
    # Determinista: segunda ejecución produce la misma asignación.
    report2 = make_splits(manifest, out_path=tmp_path / "splits2.json")
    doc1 = json.loads((tmp_path / "splits.json").read_text(encoding="utf-8"))
    doc2 = json.loads((tmp_path / "splits2.json").read_text(encoding="utf-8"))
    assert doc1["signer_ids"] == doc2["signer_ids"]
    assert report2.incidents == report.incidents


# --- D09 (fallo deliberado) ------------------------------------------------
def test_D09_identity_leak_detected(tmp_path):
    """Dos personas distintas comparten la MISMA secuencia -> integridad (exit 3).

    Fallo deliberado: la misma ``sequence_sha256`` repartida entre splits debe
    disparar ``IntegrityError``. Si el control de duplicados no la detecta, no se
    lanza la excepción y la prueba falla.
    """
    shared = _unique_shas("leak-shared")
    records = []
    # Suficientes personas para que train y test sean no vacíos.
    for s in range(10):
        signer = f"S{s:03d}"
        for c in range(10):
            records.append(_record(signer, c, _unique_shas(f"{signer}-{c}")))
    # Inyectar la copia cruzada: dos signers con class 0 comparten `shared`.
    records.append(_record("S000", 0, shared))
    records.append(_record("S009", 0, shared))
    manifest = tmp_path / "manifest.jsonl"
    _write_manifest(manifest, records)

    with pytest.raises(IntegrityError):
        make_splits(manifest, out_path=tmp_path / "splits.json")


# --- D10 -------------------------------------------------------------------
def test_D10_missing_class_incident_with_counts(tmp_path):
    """Falta una clase en val -> incidente con conteos; seed no se recomputa."""
    # 10 personas, pero NINGUNA aporta la clase 9 -> cobertura incompleta.
    records = []
    for s in range(10):
        signer = f"S{s:03d}"
        for c in range(9):  # clases 0..8 solamente
            records.append(_record(signer, c, _unique_shas(f"{signer}-{c}")))
    manifest = tmp_path / "manifest.jsonl"
    _write_manifest(manifest, records)

    report = make_splits(manifest, seed=20261022, out_path=tmp_path / "splits.json")
    joined = " ".join(report.incidents)
    assert "faltan clases" in joined
    assert "[9]" in joined
    # Los incidentes incluyen conteos (no están vacíos) y la seed es la canónica.
    doc = json.loads((tmp_path / "splits.json").read_text(encoding="utf-8"))
    assert doc["seed"] == 20261022
    assert "conteos=" in joined
