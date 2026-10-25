"""Splits deterministas sin fuga (design §10).

``make_splits(manifest_path, seed=20261022) -> SplitReport``:

1. Extraer ``signer_id`` únicos y ordenarlos de forma estable.
2. ``rng = numpy.random.default_rng(seed); perm = rng.permutation(ids)``.
3. ``n_test = ceil(0.20*N)``,
   ``n_val = max(1, int(floor(0.10*N + 0.5)))`` (HALF-UP determinista, NO el
   ``round()`` nativo half-to-even), ``n_train = N - n_test - n_val``.
   Para N=70: 49/7/14.
4. Persistir la asignación concreta en ``splits.json``; NO recomputar al consumir.
5. Controles: intersección vacía de identidades y de ``sample_ids``, y ausencia
   de duplicados por ``sequence_sha256`` -> si fallan, ``IntegrityError`` (exit 3).
6. Cobertura: 10 clases por split; si falta alguna, se emite un incidente con
   conteos y NO se recomputa la semilla.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from contracts.canonical import manifest_sha256 as _manifest_sha256
from contracts.models import NUM_CLASSES, SplitReport

from .errors import IntegrityError

__all__ = [
    "make_splits",
    "split_counts",
]

SEED_DEFAULT = 20261022
SPLITS_SCHEMA_VERSION = "lsc.splits.v1"
SPLITS_VERSION = "splits_v1"


def split_counts(n: int) -> tuple[int, int, int]:
    """Devuelve ``(n_train, n_val, n_test)`` con el redondeo HALF-UP de §10.

    ``n_val`` usa ``floor(0.10*N + 0.5)`` (half-up) y ``max(1, ...)``; nunca
    ``round()`` nativo. Para N=70 -> (49, 7, 14).
    """
    n_test = math.ceil(0.20 * n)
    n_val = max(1, int(math.floor(0.10 * n + 0.5)))
    n_train = n - n_test - n_val
    return n_train, n_val, n_test


def _load_manifest_records(manifest_path: str | Path) -> list[dict]:
    path = Path(manifest_path)
    if not path.is_file():
        raise IntegrityError(f"manifiesto no encontrado: {path}")
    records: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def make_splits(
    manifest_path: str | Path,
    seed: int = SEED_DEFAULT,
    *,
    out_path: str | Path | None = None,
) -> SplitReport:
    """Produce splits deterministas por persona y persiste ``splits.json``.

    Si ``out_path`` es ``None`` se escribe junto al manifiesto como
    ``splits.json``. Devuelve un ``SplitReport``; los controles de fuga lanzan
    ``IntegrityError`` (exit 3).
    """
    import numpy as np  # import perezoso: numpy está instalado, no en import del paquete

    records = _load_manifest_records(manifest_path)

    # signer_id -> conjunto de (sample_id, class_id, sequence_sha256)
    by_signer: dict[str, list[dict]] = {}
    for rec in records:
        by_signer.setdefault(rec["signer_id"], []).append(rec)

    signer_ids = sorted(by_signer)
    n = len(signer_ids)
    if n == 0:
        raise IntegrityError("manifiesto sin muestras; no hay signer_id para dividir")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(np.array(signer_ids, dtype=object))
    permuted = [str(s) for s in perm.tolist()]

    n_train, n_val, n_test = split_counts(n)
    if n_train < 0:
        raise IntegrityError(
            f"N={n} demasiado pequeño para train/val/test "
            f"({n_train}/{n_val}/{n_test})"
        )

    test_ids = sorted(permuted[:n_test])
    val_ids = sorted(permuted[n_test : n_test + n_val])
    train_ids = sorted(permuted[n_test + n_val :])

    # Control de fuga de identidad: intersección vacía (design §10.5).
    sets = {"train": set(train_ids), "val": set(val_ids), "test": set(test_ids)}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = sets[a] & sets[b]
        if overlap:
            raise IntegrityError(
                f"fuga de identidad: signer_id {sorted(overlap)} en {a} y {b}"
            )

    sample_ids_by_split: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    class_coverage: dict[str, dict[str, int]] = {
        "train": {},
        "val": {},
        "test": {},
    }
    seq_seen: dict[str, str] = {}
    for split_name, ids in (("train", train_ids), ("val", val_ids), ("test", test_ids)):
        for signer_id in ids:
            for rec in by_signer[signer_id]:
                sample_ids_by_split[split_name].append(rec["sample_id"])
                class_id = rec.get("class_id")
                if class_id is not None:
                    key = str(class_id)
                    cov = class_coverage[split_name]
                    cov[key] = cov.get(key, 0) + 1
                seq_hash = rec["sequence_sha256"]
                if seq_hash in seq_seen and seq_seen[seq_hash] != rec["sample_id"]:
                    raise IntegrityError(
                        f"duplicado por sequence_sha256={seq_hash}: "
                        f"sample_id {seq_seen[seq_hash]} y {rec['sample_id']}"
                    )
                seq_seen.setdefault(seq_hash, rec["sample_id"])

    # Control de fuga de sample_ids: intersección vacía entre splits.
    id_sets = {k: set(v) for k, v in sample_ids_by_split.items()}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = id_sets[a] & id_sets[b]
        if overlap:
            raise IntegrityError(
                f"fuga de muestras: sample_id {sorted(overlap)} en {a} y {b}"
            )

    # Cobertura de clases: incidente (no fatal) si falta alguna clase por split.
    incidents: list[str] = []
    for split_name in ("train", "val", "test"):
        present = set(class_coverage[split_name])
        missing = sorted(
            c for c in range(NUM_CLASSES) if str(c) not in present
        )
        if missing:
            incidents.append(
                f"cobertura incompleta en {split_name}: faltan clases {missing}; "
                f"conteos={dict(sorted(class_coverage[split_name].items()))}"
            )

    manifest_hash = _manifest_sha256(records)

    splits_doc = {
        "schema_version": SPLITS_SCHEMA_VERSION,
        "seed": seed,
        "manifest_sha256": manifest_hash,
        "version": SPLITS_VERSION,
        "signer_ids": {"train": train_ids, "val": val_ids, "test": test_ids},
        "sample_ids": {
            "train": sorted(sample_ids_by_split["train"]),
            "val": sorted(sample_ids_by_split["val"]),
            "test": sorted(sample_ids_by_split["test"]),
        },
        "class_coverage": class_coverage,
        "incidents": incidents,
    }

    target = Path(out_path) if out_path is not None else Path(manifest_path).with_name("splits.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(splits_doc, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    return SplitReport(
        n_train=len(train_ids),
        n_val=len(val_ids),
        n_test=len(test_ids),
        n_signers=n,
        class_coverage=class_coverage,
        incidents=incidents,
        splits_path=str(target),
    )
