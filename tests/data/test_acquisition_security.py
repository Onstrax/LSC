"""Pruebas de adquisición segura D12 (design §7, §20).

Archivo con rutas peligrosas (``..`` / absoluta / ejecutable / symlink) ->
rechazo, nada se mueve a destino, ``IntegrityError`` (exit 3). ``source.yaml``
mal formado -> ``ConfigError`` (exit 2). Sin estado parcial (temp dir).
"""

from __future__ import annotations

import zipfile

import pytest

from src.lsc_data.errors import ConfigError, IntegrityError
from src.lsc_data.security import is_safe_member_name, safe_extract


def _make_zip(path, members: list[tuple[str, bytes]]):
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in members:
            zf.writestr(name, data)


def test_D12_traversal_member_rejected(tmp_path):
    """Un miembro con '..' hace rechazar el archivo; destino queda vacío."""
    archive = tmp_path / "evil.zip"
    _make_zip(archive, [("ok/frame_0.jpg", b"x"), ("../escape.jpg", b"y")])
    dest = tmp_path / "dest"
    with pytest.raises(IntegrityError):
        safe_extract(archive, dest)
    # Nada se movió a destino (sin estado parcial).
    assert not dest.exists() or not any(dest.rglob("*"))


def test_D12_absolute_member_rejected(tmp_path):
    archive = tmp_path / "abs.zip"
    _make_zip(archive, [("/etc/passwd", b"y"), ("ok/frame.jpg", b"x")])
    dest = tmp_path / "dest"
    with pytest.raises(IntegrityError):
        safe_extract(archive, dest)
    assert not dest.exists() or not any(dest.rglob("*"))


def test_D12_executable_extension_rejected(tmp_path):
    """Extensión ejecutable inesperada -> rechazo completo del archivo."""
    archive = tmp_path / "exe.zip"
    _make_zip(archive, [("payload.exe", b"MZ"), ("ok/frame.jpg", b"x")])
    dest = tmp_path / "dest"
    with pytest.raises(IntegrityError):
        safe_extract(archive, dest, allowed_extensions=[".jpg"])
    assert not dest.exists() or not any(dest.rglob("*"))


def test_D12_symlink_member_rejected(tmp_path):
    """Un miembro symlink (external_attr UNIX) -> rechazo."""
    archive = tmp_path / "sym.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        info = zipfile.ZipInfo("link.jpg")
        # bits de modo UNIX: symlink 0o120000 en los 16 bits altos.
        info.external_attr = (0o120777 & 0xFFFF) << 16
        zf.writestr(info, b"/etc/passwd")
    dest = tmp_path / "dest"
    with pytest.raises(IntegrityError):
        safe_extract(archive, dest)
    assert not dest.exists() or not any(dest.rglob("*"))


def test_D12_safe_archive_extracts(tmp_path):
    """Un archivo limpio sí se extrae a destino (control positivo)."""
    archive = tmp_path / "clean.zip"
    _make_zip(archive, [("a/frame_0.jpg", b"x"), ("a/frame_1.jpg", b"y")])
    dest = tmp_path / "dest"
    moved = safe_extract(archive, dest, allowed_extensions=[".jpg"])
    assert sorted(moved) == ["a/frame_0.jpg", "a/frame_1.jpg"]
    assert (dest / "a" / "frame_0.jpg").is_file()


def test_D12_malformed_source_yaml_config_error(tmp_path):
    """source.yaml mal formado -> ConfigError (exit 2), sin estado parcial."""
    yaml = pytest.importorskip("yaml")  # noqa: F841 - PyYAML requerido para este camino
    from src.lsc_data.acquisition import load_source_config

    bad = tmp_path / "source.yaml"
    bad.write_text("name: [unterminated\n  : : :", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_source_config(bad)


def test_is_safe_member_name_unit():
    assert is_safe_member_name("a/b/c.jpg")
    assert not is_safe_member_name("/abs.jpg")
    assert not is_safe_member_name("../up.jpg")
    assert not is_safe_member_name("C:/win.jpg")
