"""El conjunto de reason_codes de Ortiz == frozenset de la API (design §18, §20).

Comparación de CONJUNTOS (orden irrelevante). La fuente de referencia es
``api/constants.QUALITY_REASON_CODES`` (sólo lectura).
"""

from __future__ import annotations

from api.constants import QUALITY_REASON_CODES as API_CODES
from src.lsc_features.reason_codes import QUALITY_REASON_CODES as ORTIZ_CODES


def test_reason_codes_sets_identical():
    assert set(ORTIZ_CODES) == set(API_CODES)


def test_reason_codes_count_is_nine():
    assert len(set(ORTIZ_CODES)) == 9
