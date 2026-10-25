"""Fixtures del módulo de datos (design §20).

Construye frames sintéticos (``FrameInput``) y helpers para fabricar secuencias
reutilizando ``contracts.models`` y los fixtures sintéticos de
``contracts/fixtures``. Ninguna prueba toca el corpus real, red, ni ``.task``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.lsc_data.adapter_lsc70w import ParsedFrame
from src.lsc_data.sequence import FrameInput

CONTRACTS_FIXTURES = Path(__file__).resolve().parents[2] / "contracts" / "fixtures"

SOURCE = "SYNTHETIC"
SOURCE_VERSION = "SYNTHETIC-v0"
# label_map mínimo del vocabulario (code -> class_id), orden canónico.
LABEL_MAP = {
    "hola": 0,
    "buenos": 1,
    "dias": 2,
    "tardes": 3,
    "noches": 4,
    "yo": 5,
    "nombre": 6,
    "anos": 7,
    "gustar": 8,
    "licor": 9,
}


def _frame_hash(signer_id: str, class_id: int, repetition_id: int, idx: int) -> str:
    """Hash sintético estable de 64 hex por frame (sin leer imágenes)."""
    seed = f"{signer_id}|{class_id}|{repetition_id}|{idx}"
    # Determinista y legible; no es un hash de archivo real, sólo un marcador.
    return ("%064x" % (abs(hash(seed)) % (16 ** 64)))


def make_frame(
    *,
    signer_id: str = "S000",
    class_label: str = "hola",
    class_id: int = 0,
    repetition_id: int = 0,
    frame_index: int,
    frame_sha256: str | None = None,
    width: int = 640,
    height: int = 480,
) -> FrameInput:
    """Construye un ``FrameInput`` sintético con gramática ya interpretada."""
    path = f"lsc70_v2/{signer_id}/{class_label}/{repetition_id}/frame_{frame_index}.jpg"
    parsed = ParsedFrame(
        signer_id=signer_id,
        class_label=class_label,
        class_id=class_id,
        repetition_id=repetition_id,
        frame_index=frame_index,
        frame_path=path,
    )
    sha = frame_sha256 or _frame_hash(signer_id, class_id, repetition_id, frame_index)
    return FrameInput(parsed=parsed, frame_sha256=sha, width=width, height=height)


def make_sequence_frames(
    *,
    signer_id: str = "S000",
    class_label: str = "hola",
    class_id: int = 0,
    repetition_id: int = 0,
    indices=range(6),
    shas: list[str] | None = None,
    width: int = 640,
    height: int = 480,
) -> list[FrameInput]:
    """Seis frames (o los ``indices`` dados) de una misma muestra."""
    indices = list(indices)
    frames: list[FrameInput] = []
    for pos, idx in enumerate(indices):
        sha = shas[pos] if shas is not None else None
        frames.append(
            make_frame(
                signer_id=signer_id,
                class_label=class_label,
                class_id=class_id,
                repetition_id=repetition_id,
                frame_index=idx,
                frame_sha256=sha,
                width=width,
                height=height,
            )
        )
    return frames


@pytest.fixture
def frame_factory():
    return make_frame


@pytest.fixture
def sequence_factory():
    return make_sequence_frames


@pytest.fixture
def label_map():
    return dict(LABEL_MAP)


@pytest.fixture
def contracts_fixtures_dir() -> Path:
    return CONTRACTS_FIXTURES
