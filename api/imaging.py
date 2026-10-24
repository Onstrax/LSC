"""Decodificación y validación de JPEG a RGB.

Reglas (SDD §9, §11):
- Verificar que el JPEG es decodificable; el nombre y el Content-Type no bastan.
- Comprobar el tamaño real antes de asignar grandes buffers.
- No aumentar datos, interpolar ni normalizar en la API.
- Las seis imágenes deben tener dimensiones idénticas.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image, UnidentifiedImageError

from api.constants import MAX_HEIGHT, MAX_PIXELS, MAX_WIDTH


class ImageError(ValueError):
    """JPEG inválido o con dimensiones fuera de límite."""

    def __init__(self, message: str, *, reason_code: str = "INVALID_IMAGE") -> None:
        super().__init__(message)
        self.reason_code = reason_code


def decode_jpeg_to_rgb(data: bytes) -> np.ndarray:
    """Decodifica bytes JPEG a una matriz RGB uint8 (H, W, 3).

    Lanza :class:`ImageError` si el formato no es JPEG, si no es decodificable o
    si excede los límites de tamaño.
    """
    try:
        with Image.open(io.BytesIO(data)) as img:
            fmt = (img.format or "").upper()
            if fmt not in {"JPEG", "JPG"}:
                raise ImageError(
                    f"Formato {fmt or 'desconocido'} no admitido; se requiere JPEG.",
                    reason_code="UNSUPPORTED_MEDIA_TYPE",
                )
            width, height = img.size
            if width <= 0 or height <= 0:
                raise ImageError("Dimensiones de imagen inválidas.")
            if width > MAX_WIDTH or height > MAX_HEIGHT:
                raise ImageError(
                    f"Imagen {width}x{height} excede {MAX_WIDTH}x{MAX_HEIGHT}.",
                    reason_code="CONTRACT_VIOLATION",
                )
            if width * height > MAX_PIXELS:
                raise ImageError(
                    f"Imagen con {width * height} px excede {MAX_PIXELS}.",
                    reason_code="CONTRACT_VIOLATION",
                )
            rgb = img.convert("RGB")
            arr = np.asarray(rgb, dtype=np.uint8)
    except UnidentifiedImageError as exc:
        raise ImageError("No se pudo decodificar la imagen (JPEG inválido).") from exc
    except OSError as exc:
        raise ImageError("Imagen JPEG corrupta o truncada.") from exc

    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ImageError("La imagen decodificada no es RGB de tres canales.")
    return arr


def decode_sequence(frames: list[bytes], *, declared_wh: tuple[int, int]) -> list[np.ndarray]:
    """Decodifica seis JPEG y verifica que comparten dimensiones.

    `declared_wh` es (width, height) de la metadata; debe coincidir con las
    dimensiones reales decodificadas.
    """
    decoded: list[np.ndarray] = []
    dwidth, dheight = declared_wh
    for i, data in enumerate(frames):
        arr = decode_jpeg_to_rgb(data)
        h, w = arr.shape[0], arr.shape[1]
        if (w, h) != (dwidth, dheight):
            raise ImageError(
                f"frame_{i}: dimensiones {w}x{h} != metadata {dwidth}x{dheight}.",
                reason_code="DIMENSION_MISMATCH",
            )
        if decoded and arr.shape != decoded[0].shape:
            raise ImageError(
                f"frame_{i}: dimensiones distintas a frame_0.",
                reason_code="SIZE_MISMATCH",
            )
        decoded.append(arr)
    return decoded
