"""Inventario del corpus crudo (design §19).

``inventory`` recorre un directorio de datos crudos y escribe un CSV con el
conteo de archivos por carpeta de primer nivel (p. ej. por ``signer_id``). No
interpreta la gramática ni lee píxeles: es un censo barato para saber qué hay
antes de construir el manifiesto. Las salidas (``reports/data/inventory.csv``)
están git-ignoradas.
"""

from __future__ import annotations

import csv
from pathlib import Path

from .errors import ConfigError

__all__ = ["inventory", "write_inventory"]

INVENTORY_HEADER = ("group", "n_files", "n_subdirs")


def inventory(raw_dir: str | Path, out_path: str | Path) -> list[tuple[str, int, int]]:
    """Censa ``raw_dir`` por carpeta de primer nivel y escribe ``out_path``.

    Devuelve las filas ``(group, n_files, n_subdirs)`` ordenadas por ``group``.
    ``raw_dir`` inexistente es ``ConfigError`` (exit 2).
    """
    root = Path(raw_dir)
    if not root.is_dir():
        raise ConfigError(f"raw-dir no es un directorio: {root}")

    rows: list[tuple[str, int, int]] = []
    for group_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        files = [p for p in group_dir.rglob("*") if p.is_file()]
        subdirs = [p for p in group_dir.rglob("*") if p.is_dir()]
        rows.append((group_dir.name, len(files), len(subdirs)))

    write_inventory(rows, out_path)
    return rows


def write_inventory(
    rows: list[tuple[str, int, int]], out_path: str | Path
) -> None:
    """Escribe el inventario como CSV con encabezado fijo."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(INVENTORY_HEADER)
        for row in rows:
            writer.writerow(row)
