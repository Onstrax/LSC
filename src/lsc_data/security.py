"""Extracción segura de archivos (design §7).

Flujo:

1. Extraer el archivo a un directorio TEMPORAL (nunca directo al destino).
2. Rechazar rutas peligrosas: absolutas, componentes ``..``, symlinks que
   escapan del destino, y ejecutables inesperados (extensiones no permitidas por
   config). Cualquier rechazo -> ``IntegrityError`` (exit 3) sin dejar estado
   parcial (el temp dir se descarta).
3. Sólo tras validar TODO el contenido, mover a destino.
4. Mismo input (mismo hash de miembro) no se duplica; input distinto NO
   sobrescribe (se requiere nueva versión de procedencia en ``source.yaml``).

Soporta ``.zip`` y archivos tar (``.tar``, ``.tar.gz``/``.tgz``). No descarga
nada por red.
"""

from __future__ import annotations

import hashlib
import shutil
import tarfile
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Iterable

from .errors import IntegrityError

__all__ = [
    "safe_extract",
    "is_safe_member_name",
]

_DEFAULT_ALLOWED = frozenset(
    {".jpg", ".jpeg", ".png", ".txt", ".csv", ".json"}
)


def is_safe_member_name(name: str) -> bool:
    """Verdadero si el nombre de miembro no es absoluto ni tiene ``..``."""
    normalized = name.replace("\\", "/")
    if not normalized or normalized.startswith("/"):
        return False
    if len(normalized) >= 2 and normalized[1] == ":":  # unidad de Windows
        return False
    parts = PurePosixPath(normalized).parts
    return ".." not in parts


def _check_member(
    name: str, *, is_symlink: bool, allowed_ext: frozenset[str]
) -> None:
    """Valida un nombre de miembro; lanza ``IntegrityError`` si es peligroso."""
    if is_symlink:
        raise IntegrityError(f"symlink no permitido en el archivo: {name!r}")
    if not is_safe_member_name(name):
        raise IntegrityError(f"ruta peligrosa en el archivo: {name!r}")
    suffix = PurePosixPath(name.replace("\\", "/")).suffix.lower()
    # Las entradas de directorio (sin sufijo y terminadas en '/') se permiten.
    if name.replace("\\", "/").endswith("/"):
        return
    if suffix and suffix not in allowed_ext:
        raise IntegrityError(
            f"extensión inesperada {suffix!r} en {name!r}; "
            f"permitidas: {sorted(allowed_ext)}"
        )


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _iter_zip_members(zf: zipfile.ZipFile) -> Iterable[tuple[str, bool]]:
    for info in zf.infolist():
        # bit alto 0xA000 de external_attr indica symlink en UNIX.
        mode = (info.external_attr >> 16) & 0o170000
        is_symlink = mode == 0o120000
        yield info.filename, is_symlink


def safe_extract(
    archive_path: str | Path,
    dest_dir: str | Path,
    *,
    allowed_extensions: Iterable[str] | None = None,
) -> list[str]:
    """Extrae ``archive_path`` a ``dest_dir`` de forma segura (design §7).

    Devuelve la lista de rutas relativas movidas a destino (ordenada). Rutas
    peligrosas -> ``IntegrityError`` (exit 3) y NADA se mueve. Un miembro cuyo
    contenido ya existe con el MISMO hash en destino no se duplica; si existe con
    hash DISTINTO se lanza ``IntegrityError`` (no sobrescribe).
    """
    archive = Path(archive_path)
    if not archive.is_file():
        raise IntegrityError(f"archivo no encontrado: {archive}")

    allowed = (
        frozenset(e.lower() for e in allowed_extensions)
        if allowed_extensions is not None
        else _DEFAULT_ALLOWED
    )
    destination = Path(dest_dir)

    with tempfile.TemporaryDirectory(prefix="lsc_extract_") as tmp:
        tmp_root = Path(tmp)
        extracted: list[str] = []

        if zipfile.is_zipfile(archive):
            with zipfile.ZipFile(archive) as zf:
                for name, is_symlink in _iter_zip_members(zf):
                    _check_member(name, is_symlink=is_symlink, allowed_ext=allowed)
                for info in zf.infolist():
                    if info.filename.endswith("/"):
                        continue
                    target = tmp_root / info.filename
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(info) as src, target.open("wb") as dst:
                        shutil.copyfileobj(src, dst)
                    extracted.append(info.filename.replace("\\", "/"))
        elif tarfile.is_tarfile(archive):
            with tarfile.open(archive) as tf:
                for member in tf.getmembers():
                    _check_member(
                        member.name,
                        is_symlink=member.issym() or member.islnk(),
                        allowed_ext=allowed,
                    )
                for member in tf.getmembers():
                    if not member.isfile():
                        continue
                    extracted_file = tf.extractfile(member)
                    if extracted_file is None:
                        continue
                    target = tmp_root / member.name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with target.open("wb") as dst:
                        shutil.copyfileobj(extracted_file, dst)
                    extracted.append(member.name.replace("\\", "/"))
        else:
            raise IntegrityError(
                f"formato de archivo no soportado (ni zip ni tar): {archive}"
            )

        # Defensa en profundidad: ningún archivo extraído escapa del temp root.
        tmp_resolved = tmp_root.resolve()
        for rel in extracted:
            resolved = (tmp_root / rel).resolve()
            if tmp_resolved not in resolved.parents and resolved != tmp_resolved:
                raise IntegrityError(f"miembro escapa del temp dir: {rel!r}")

        # Sólo tras validar TODO, mover a destino con reglas de duplicado.
        destination.mkdir(parents=True, exist_ok=True)
        moved: list[str] = []
        for rel in sorted(extracted):
            src = tmp_root / rel
            dst = destination / rel
            new_hash = _sha256_bytes(src.read_bytes())
            if dst.exists():
                existing_hash = _sha256_bytes(dst.read_bytes())
                if existing_hash == new_hash:
                    continue  # mismo input: no se duplica
                raise IntegrityError(
                    f"input distinto no sobrescribe {rel!r}; requiere nueva "
                    "versión de procedencia en source.yaml"
                )
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
            moved.append(rel)

    return sorted(moved)
