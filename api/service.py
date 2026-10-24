"""Estado del servicio: construcción de adaptadores y worker según el modo.

Mantiene el estado compartido (worker, etiquetas, versiones, modo) que las
rutas consultan mediante inyección de dependencias. Permite reemplazar los
componentes por dobles en pruebas.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from api.config import Settings, load_labels
from api.constants import MODE_MOCK
from api.worker import InferenceWorker


@dataclass
class ServiceState:
    """Estado vivo de la aplicación durante su ciclo de vida."""

    settings: Settings
    worker: InferenceWorker | None = None
    labels: list[str] = field(default_factory=list)
    mode: str = ""
    # Diagnóstico para el operador cuando el servicio no está disponible.
    unavailable_reason: str | None = None

    @property
    def mode_label(self) -> str:
        return self.mode or self.settings.mode


def build_state(settings: Settings) -> ServiceState:
    """Construye el estado SIN cargar pesos todavía (eso ocurre en lifespan)."""
    return ServiceState(settings=settings, mode=settings.mode)


def initialize_worker(state: ServiceState) -> None:
    """Carga adaptadores y verifica compatibilidad. Publica readiness o 503.

    En modo real, si faltan artefactos o la política es incompatible, deja el
    worker en estado no listo con un diagnóstico; NUNCA cambia a mock.
    """
    settings = state.settings
    if settings.mode == MODE_MOCK:
        from api.adapters.mock import MockDecisionAdapter, MockExtractor, MockPredictor

        extractor = MockExtractor()
        predictor = MockPredictor()
        decision = MockDecisionAdapter(threshold=settings.mock_threshold)
        state.labels = load_labels(settings)
    else:
        try:
            from api.adapters.real import (
                RealDecisionAdapter,
                RealExtractor,
                RealPredictor,
            )

            extractor = RealExtractor(settings.artifacts_dir)
            predictor = RealPredictor(settings.artifacts_dir)
            decision = RealDecisionAdapter(settings.artifacts_dir)
            state.labels = load_labels(settings)
        except Exception as exc:
            # Servicio no disponible: health responderá 503 con diagnóstico.
            state.unavailable_reason = str(exc)
            state.worker = None
            return

    worker = InferenceWorker(
        extractor=extractor,
        predictor=predictor,
        decision=decision,
        job_timeout_ms=settings.server_job_timeout_ms,
    )
    try:
        worker.verify_compatibility()
    except Exception as exc:
        state.unavailable_reason = str(exc)
        worker.shutdown()
        state.worker = None
        return

    worker.mark_ready()
    state.worker = worker
    state.unavailable_reason = None
