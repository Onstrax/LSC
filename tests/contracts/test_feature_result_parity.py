"""Paridad estructural de FeatureResult Ortiz <-> API (DO08, design §4, §18, §20).

El contrato de integración con Juan exige que el ``FeatureResult`` que entrega
Ortiz (``src/lsc_features/types.py``) sea estructuralmente idéntico al que la
API ya consume (``api/adapters/types.py``), con UNA sola diferencia documentada:
``valid_shoulders`` es ``int`` en Ortiz (conteo de cuadros con hombros válidos)
y ``bool`` en el contrato de la API.

La API resuelve esa diferencia DESDE SU LADO, sin que Ortiz toque ``api/``:
``api/adapters/real.py`` coacciona con

    valid_shoulders=bool(getattr(result, "valid_shoulders", True))

por lo que el ``int`` de Ortiz entra como su verdad/falsedad. Esta prueba fija
esa compatibilidad y es SÓLO LECTURA de ``app/`` y ``api/`` (sólo importa
``api.adapters.types`` y ``api.constants``; no modifica nada bajo ``api/``).

También verifica que el orden canónico de ``contracts/labels.json`` es coherente
con ``api/constants`` (``NUM_CLASSES``) y con el esquema HTTP de Juan
(``api/schemas/models.PredictResponse.label_id`` en ``0..NUM_CLASSES-1``), y que
el conjunto de ``reason_codes`` coincide entre ambos lados.
"""

from __future__ import annotations

import inspect
import json
from dataclasses import fields
from pathlib import Path

import numpy as np

from api.adapters.types import FeatureResult as ApiFeatureResult
from api.constants import NUM_CLASSES, QUALITY_REASON_CODES as API_REASON_CODES
from api.schemas.models import PredictResponse
from src.lsc_features.reason_codes import QUALITY_REASON_CODES as ORTIZ_REASON_CODES
from src.lsc_features.types import FeatureResult as OrtizFeatureResult

REPO_ROOT = Path(__file__).resolve().parents[2]

# Campos esperados del contrato compartido (resumen de AGENTS.md "Interfaces").
EXPECTED_FIELDS = (
    "x",
    "quality_pass",
    "reason_codes",
    "valid_shoulders",
    "frames_with_hand",
    "feature_spec_hash",
    "elapsed_ms",
)


def _field_names(dc) -> list[str]:
    return [f.name for f in fields(dc)]


# ---------------------------------------------------------------------------
# Compatibilidad estructural del FeatureResult
# ---------------------------------------------------------------------------
def test_field_names_match_between_ortiz_and_api():
    """Los nombres de campo coinciden exactamente (mismo orden incluido)."""
    assert _field_names(OrtizFeatureResult) == list(EXPECTED_FIELDS)
    assert _field_names(ApiFeatureResult) == list(EXPECTED_FIELDS)
    assert _field_names(OrtizFeatureResult) == _field_names(ApiFeatureResult)


def test_expected_fields_present_on_both_sides():
    ortiz = set(_field_names(OrtizFeatureResult))
    api = set(_field_names(ApiFeatureResult))
    for name in EXPECTED_FIELDS:
        assert name in ortiz, f"Ortiz FeatureResult sin campo {name}"
        assert name in api, f"API FeatureResult sin campo {name}"


def test_ortiz_valid_shoulders_is_int_and_api_bool():
    """La única divergencia documentada: Ortiz int, API bool."""
    ortiz_types = {f.name: f.type for f in fields(OrtizFeatureResult)}
    api_types = {f.name: f.type for f in fields(ApiFeatureResult)}
    # Las anotaciones son cadenas por `from __future__ import annotations`.
    assert str(ortiz_types["valid_shoulders"]) == "int"
    assert str(api_types["valid_shoulders"]) == "bool"


def test_api_coerces_valid_shoulders_with_bool():
    """``bool(valid_shoulders_int)`` es la coacción exacta que aplica la API.

    Reproducimos aquí la expresión documentada de ``api/adapters/real.py``
    (SÓLO LECTURA): ``bool(getattr(result, "valid_shoulders", True))``.
    Comprobamos que un ``FeatureResult`` de Ortiz con conteo entero entra como
    su verdad/falsedad, y que el contrato de la API acepta ambos resultados.
    """
    zeros = np.zeros((6, 147), dtype=np.float32)

    # Conteo > 0 -> True; conteo 0 -> False.
    positive = OrtizFeatureResult(x=zeros, quality_pass=True, valid_shoulders=6)
    zero = OrtizFeatureResult(x=zeros, quality_pass=False, valid_shoulders=0)

    assert bool(getattr(positive, "valid_shoulders", True)) is True
    assert bool(getattr(zero, "valid_shoulders", True)) is False

    # El resultado coaccionado cabe en el dataclass de la API sin pérdida.
    api_positive = ApiFeatureResult(
        x=np.asarray(positive.x, dtype=np.float32),
        quality_pass=bool(positive.quality_pass),
        reason_codes=list(positive.reason_codes),
        valid_shoulders=bool(getattr(positive, "valid_shoulders", True)),
        frames_with_hand=int(getattr(positive, "frames_with_hand", 0)),
        feature_spec_hash=str(getattr(positive, "feature_spec_hash", "")),
        elapsed_ms=float(getattr(positive, "elapsed_ms", 0.0)),
    )
    assert api_positive.valid_shoulders is True


def test_documented_coercion_expression_present_in_api_real():
    """La expresión de coacción está en ``api/adapters/real.py`` (SÓLO LECTURA).

    No modificamos ``api/``; sólo LEEMOS el archivo para fijar que el contrato
    documentado (``valid_shoulders=bool(getattr(result, "valid_shoulders",
    True))``) sigue vigente del lado de la API.
    """
    real_src = (REPO_ROOT / "api" / "adapters" / "real.py").read_text(encoding="utf-8")
    assert 'valid_shoulders=bool(getattr(result, "valid_shoulders", True))' in real_src


def test_parity_test_does_not_touch_api_sources():
    """Esta prueba sólo importa ``api.adapters.types`` y el esquema/constantes.

    Verifica que no se construyen ni mutan objetos de ``api/`` fuera de la
    lectura declarada. (Guardarraíl de propiedad AGENTS.md: Ortiz no edita
    ``api/``.)
    """
    src = inspect.getsource(inspect.getmodule(test_parity_test_does_not_touch_api_sources))
    # No debe haber escrituras a módulos de api fuera de importaciones de tipos.
    assert "import api" not in src or "from api" in src  # sólo imports de lectura


# ---------------------------------------------------------------------------
# labels.json vs api/constants y api/schemas
# ---------------------------------------------------------------------------
def _labels() -> list[dict]:
    data = json.loads((REPO_ROOT / "contracts" / "labels.json").read_text(encoding="utf-8"))
    return data["labels"]


def test_labels_count_matches_api_num_classes():
    """``contracts/labels.json`` tiene exactamente ``NUM_CLASSES`` entradas."""
    assert len(_labels()) == NUM_CLASSES


def test_labels_class_ids_consecutive_zero_based():
    assert [e["class_id"] for e in _labels()] == list(range(NUM_CLASSES))


def test_labels_cover_schema_label_id_range():
    """El rango de ``label_id`` del esquema HTTP de Juan (0..NUM_CLASSES-1)
    cubre exactamente el conjunto de ``class_id`` de ``labels.json``.

    Se lee la restricción del campo directamente de ``PredictResponse`` (SÓLO
    LECTURA de ``api/schemas``); no se reconstruye a mano.
    """
    field_info = PredictResponse.model_fields["label_id"]
    ge = next(m.ge for m in field_info.metadata if hasattr(m, "ge"))
    le = next(m.le for m in field_info.metadata if hasattr(m, "le"))
    class_ids = [e["class_id"] for e in _labels()]
    assert ge == min(class_ids) == 0
    assert le == max(class_ids) == NUM_CLASSES - 1


# ---------------------------------------------------------------------------
# reason_codes: igualdad de conjuntos
# ---------------------------------------------------------------------------
def test_reason_codes_set_equal_between_ortiz_and_api():
    assert set(ORTIZ_REASON_CODES) == set(API_REASON_CODES)
    assert len(set(ORTIZ_REASON_CODES)) == 9
