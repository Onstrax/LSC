"""Lectura del bundle de características y guardia del conjunto de test.

Reglas (plan §2, §6):
- Se entrena SOLO con muestras procesables (``quality_pass``) del split ``train``.
- Validación se evalúa de dos modos: procesables e integral (rechazos cuentan).
- ``test`` está reservado: ``load_split('test')`` exige ``allow_test=True`` Y que
  exista ``reports/models/freeze.json``. Ningún experimento de A1–A3 lo lee.
- Etiqueta e identificadores nunca son características.
- Un bundle sintético (``synthetic_contract_fixture``) queda marcado y NUNCA es
  evidencia de exactitud.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SPLITS = ("train", "val", "test")
SEQ_SHAPE = (6, 147)
NUM_CLASSES = 10
FREEZE_FILE = Path("reports/models/freeze.json")
SYNTHETIC_FLAG = "synthetic_contract_fixture"


class TestAccessError(RuntimeError):
    """Intento de leer el test antes de congelar las decisiones."""

    __test__ = False  # que pytest no lo recolecte como clase de pruebas


class BundleError(ValueError):
    """El bundle no cumple el contrato features.npz v1."""


@dataclass(frozen=True)
class Split:
    name: str
    X: np.ndarray  # float32 (N,6,147)
    y: np.ndarray  # int64 (N,)
    sample_id: np.ndarray
    signer_id: np.ndarray
    quality_pass: np.ndarray  # bool (N,)

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def processable(self) -> "Split":
        m = self.quality_pass
        return Split(self.name, self.X[m], self.y[m], self.sample_id[m], self.signer_id[m], self.quality_pass[m])


class Bundle:
    """Bundle ``features.npz`` cargado con ``allow_pickle=False``."""

    def __init__(self, path: str | Path, *, freeze_file: str | Path = FREEZE_FILE):
        self.path = Path(path)
        self.freeze_file = Path(freeze_file)
        if not self.path.is_file():
            raise FileNotFoundError(
                f"No existe {self.path}. Ortiz entrega features.npz (artifacts/data/v1). "
                "No se inventan datos: use el doble 'synthetic_contract_fixture' SOLO para pruebas."
            )
        with np.load(str(self.path), allow_pickle=False) as d:
            self._arrays = {k: d[k] for k in d.files}
        self._validate()
        self.synthetic = bool(self._arrays.get(SYNTHETIC_FLAG, np.array(False)))
        self.sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()

    # ------------------------------------------------------------------
    def _validate(self) -> None:
        a = self._arrays
        for key in ("X", "y", "sample_id", "signer_id", "split", "quality_pass"):
            if key not in a:
                raise BundleError(f"falta el array '{key}' en {self.path}")
        X = a["X"]
        if X.dtype != np.float32 or X.ndim != 3 or X.shape[1:] != SEQ_SHAPE:
            raise BundleError(f"X debe ser float32 (N,6,147); es {X.dtype} {X.shape}")
        n = X.shape[0]
        for key in ("y", "sample_id", "signer_id", "split", "quality_pass"):
            if a[key].shape[0] != n:
                raise BundleError(f"longitud de '{key}' != N")
        if not np.all(np.isfinite(X)):
            raise BundleError("X contiene valores no finitos")
        bad = set(np.unique(a["split"]).tolist()) - set(SPLITS)
        if bad:
            raise BundleError(f"valores de split desconocidos: {bad}")
        # Cero fuga: ninguna persona en dos particiones.
        by_signer: dict[str, set] = {}
        for s, sp in zip(a["signer_id"].tolist(), a["split"].tolist()):
            by_signer.setdefault(s, set()).add(sp)
        leaked = sorted(s for s, v in by_signer.items() if len(v) > 1)
        if leaked:
            raise BundleError(f"personas en más de una partición (fuga): {leaked[:5]}")

    # ------------------------------------------------------------------
    def load_split(self, name: str, *, allow_test: bool = False) -> Split:
        if name not in SPLITS:
            raise ValueError(f"split desconocido: {name}")
        if name == "test":
            if not allow_test:
                raise TestAccessError("El split 'test' está reservado hasta congelar las decisiones.")
            if not self.freeze_file.is_file():
                raise TestAccessError(
                    f"No existe {self.freeze_file}: las decisiones no están congeladas "
                    "(ejecute `python -m src.lsc_models freeze`)."
                )
        a = self._arrays
        m = a["split"] == name
        return Split(
            name=name,
            X=a["X"][m].astype(np.float32, copy=False),
            y=a["y"][m].astype(np.int64, copy=False),
            sample_id=a["sample_id"][m],
            signer_id=a["signer_id"][m],
            quality_pass=a["quality_pass"][m].astype(bool),
        )

    def summary(self) -> dict:
        """Conteos SIN tocar el contenido del test (solo cuenta filas)."""
        a = self._arrays
        out = {"sha256": self.sha256, "synthetic": self.synthetic, "n_total": int(a["X"].shape[0])}
        for sp in SPLITS:
            m = a["split"] == sp
            out[f"n_{sp}"] = int(m.sum())
            out[f"n_{sp}_quality_pass"] = int(a["quality_pass"][m].sum())
        return out


# ----------------------------------------------------------------------
# Doble de prueba: synthetic_contract_fixture (nunca evidencia de exactitud)
# ----------------------------------------------------------------------
def make_synthetic_contract_fixture(
    path: str | Path,
    *,
    n_signers: int = 12,
    seed: int = 20261022,
    reject_every: int = 9,
) -> Path:
    """Escribe un bundle sintético con el MISMO contrato que ``features.npz``.

    Sirve para probar interfaces y el entrenador. Cada clase tiene una trayectoria
    media distinta + ruido, así que un modelo aprende algo, pero eso NO dice nada
    sobre la exactitud en LSC: el bundle lleva ``synthetic_contract_fixture=True``.
    """
    rng = np.random.default_rng(seed)
    protos = rng.normal(0.0, 0.6, size=(NUM_CLASSES, 6, 49, 2)).astype(np.float32)
    n_train, n_val = int(n_signers * 0.65), max(1, int(n_signers * 0.15))
    rows = []
    for s in range(n_signers):
        split = "train" if s < n_train else ("val" if s < n_train + n_val else "test")
        for c in range(NUM_CLASSES):
            noise = rng.normal(0.0, 0.25, size=(6, 49, 2)).astype(np.float32)
            xy = protos[c] + noise
            x = np.zeros((6, 49, 3), dtype=np.float32)
            x[..., 0:2] = xy
            x[..., 2] = 1.0
            rows.append((x.reshape(6, 147), c, f"SYN_{s:02d}_{c}", f"SYN{s:02d}", split))
    X = np.stack([r[0] for r in rows]).astype(np.float32)
    qp = np.ones(len(rows), dtype=bool)
    qp[::reject_every] = False
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        X=X,
        y=np.array([r[1] for r in rows], dtype=np.int64),
        sample_id=np.array([r[2] for r in rows], dtype=np.str_),
        signer_id=np.array([r[3] for r in rows], dtype=np.str_),
        split=np.array([r[4] for r in rows], dtype=np.str_),
        quality_pass=qp,
        **{SYNTHETIC_FLAG: np.array(True)},
    )
    return out if out.suffix == ".npz" else out.with_suffix(".npz")
