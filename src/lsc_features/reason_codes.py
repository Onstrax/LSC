"""Códigos de razón de calidad del extractor (design §13/§18).

Fuente de referencia (SÓLO LECTURA): el ``frozenset``
``api/constants.QUALITY_REASON_CODES`` (9 códigos). Aquí se replican como
conjunto (``QUALITY_REASON_CODES``) para comparar contra la API, y como lista
ORDENADA de EMISIÓN (``EMISSION_ORDER``, design §13). El conjunto no tiene orden;
el único orden normativo es el de emisión.
"""

from __future__ import annotations

# Códigos individuales (constantes nombradas para evitar errores de tipeo).
INVALID_FRAME_COUNT = "INVALID_FRAME_COUNT"
INVALID_IMAGE = "INVALID_IMAGE"
SIZE_MISMATCH = "SIZE_MISMATCH"
MULTIPLE_PEOPLE = "MULTIPLE_PEOPLE"
INSUFFICIENT_POSE = "INSUFFICIENT_POSE"
INSUFFICIENT_HANDS = "INSUFFICIENT_HANDS"
GEOMETRY_OUTLIER = "GEOMETRY_OUTLIER"
HAND_ASSOCIATION_AMBIGUOUS = "HAND_ASSOCIATION_AMBIGUOUS"
CONFIG_MISMATCH = "CONFIG_MISMATCH"

# Orden determinista de EMISIÓN (design §13). Fuente única del orden.
EMISSION_ORDER: tuple[str, ...] = (
    INVALID_FRAME_COUNT,
    INVALID_IMAGE,
    SIZE_MISMATCH,
    MULTIPLE_PEOPLE,
    INSUFFICIENT_POSE,
    INSUFFICIENT_HANDS,
    GEOMETRY_OUTLIER,
    HAND_ASSOCIATION_AMBIGUOUS,
    CONFIG_MISMATCH,
)

# Conjunto espejo del frozenset de la API (orden irrelevante; se compara por ==).
QUALITY_REASON_CODES = frozenset(EMISSION_ORDER)

# Advertencias no bloqueantes: no implican rechazo por sí solas (design §13).
NON_BLOCKING_WARNINGS = frozenset({HAND_ASSOCIATION_AMBIGUOUS})


def order_reason_codes(codes) -> list[str]:
    """Devuelve ``codes`` únicos en el orden determinista de emisión (§13)."""
    present = set(codes)
    return [code for code in EMISSION_ORDER if code in present]
