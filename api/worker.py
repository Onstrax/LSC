"""Trabajador de inferencia dedicado y control de concurrencia.

Reglas (SDD §11, §12):
- Un solo trabajador ejecuta extractor y predictor; se crean, usan y cierran en
  el mismo hilo del trabajador (MediaPipe/TF no son seguros entre hilos).
- Un único trabajo en ejecución; ninguna cola. Un segundo intento recibe 429.
- La plaza se mantiene ocupada hasta que el trabajo termine REALMENTE; la salida
  tardía se descarta. Cancelar el HTTP no detiene la operación nativa.
- Orden obligatorio: validar, decodificar, extraer, comprobar calidad y hash,
  formar lote (1,6,147), predecir, aplicar política, serializar.
- No ejecutar el clasificador si quality_pass=false.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FutTimeout
from dataclasses import dataclass, field

import numpy as np

from api.adapters.types import (
    DecisionProtocol,
    ExtractorProtocol,
    PredictorProtocol,
)
from api.constants import NON_BLOCKING_WARNINGS, SEQUENCE_SHAPE


class ServiceBusyError(RuntimeError):
    """Hay una inferencia activa -> 429 SERVICE_BUSY."""


class InferenceTimeoutError(RuntimeError):
    """El trabajo excedió el plazo del servidor -> 504 INFERENCE_TIMEOUT."""


@dataclass
class JobResult:
    """Resultado interno del trabajo de inferencia (antes de serializar)."""

    quality_pass: bool
    accepted: bool
    label_id: int | None
    confidence: float | None
    reason_codes: list[str] = field(default_factory=list)
    feature_spec_hash: str | None = None
    model_version: str | None = None
    timings_ms: dict = field(default_factory=dict)


class InferenceWorker:
    """Coordina extractor + predictor + decisión con una sola plaza.

    Se inicializa en estado ``loading`` y publica readiness sólo tras verificar
    artefactos y política. Un error de carga deja ``ready=False`` (health 503).
    """

    def __init__(
        self,
        *,
        extractor: ExtractorProtocol,
        predictor: PredictorProtocol,
        decision: DecisionProtocol,
        job_timeout_ms: int,
    ) -> None:
        self._extractor = extractor
        self._predictor = predictor
        self._decision = decision
        self._job_timeout_s = job_timeout_ms / 1000.0
        # Un único hilo: garantiza un solo trabajo nativo a la vez.
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="lsc-infer")
        self._slot = threading.Lock()
        self._ready = False

    # -- estado ------------------------------------------------------------
    def mark_ready(self) -> None:
        self._ready = True

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def busy(self) -> bool:
        return self._slot.locked()

    @property
    def model_version(self) -> str | None:
        return getattr(self._predictor, "model_version", None)

    @property
    def feature_spec_hash(self) -> str | None:
        return getattr(self._predictor, "feature_spec_hash", None)

    def verify_compatibility(self) -> None:
        """Compara el hash del feature_spec entre extractor y predictor.

        Una incompatibilidad impide predecir (readiness 503, CONFIG_INCOMPATIBLE).
        """
        ex_hash = getattr(self._extractor, "feature_spec_hash", "")
        md_hash = getattr(self._predictor, "feature_spec_hash", "")
        if ex_hash and md_hash and ex_hash != md_hash:
            raise RuntimeError(
                f"feature_spec_hash incompatible: extractor={ex_hash} modelo={md_hash}."
            )

    # -- ejecución ---------------------------------------------------------
    def submit(self, frames_rgb: list[np.ndarray]) -> JobResult:
        """Adquiere la plaza y ejecuta el trabajo. Si está ocupada, lanza 429.

        Mantiene la plaza hasta que el trabajo termina realmente, aun si el
        plazo expira (se descarta la salida tardía, pero no se ejecuta otro).
        """
        acquired = self._slot.acquire(blocking=False)
        if not acquired:
            raise ServiceBusyError("Hay una inferencia activa.")

        future: Future = self._pool.submit(self._run_pipeline, frames_rgb)
        try:
            return future.result(timeout=self._job_timeout_s)
        except FutTimeout as exc:
            # No se detiene la operación nativa; se libera la plaza sólo cuando
            # el trabajo realmente termine (callback). Señalamos timeout.
            def _release(_f: Future) -> None:
                try:
                    self._slot.release()
                except RuntimeError:
                    pass

            future.add_done_callback(_release)
            raise InferenceTimeoutError("El trabajo de inferencia excedió el plazo.") from exc
        else:
            # Camino normal: liberar al obtener resultado.
            pass
        finally:
            # Si el future ya terminó (resultado o excepción distinta de timeout)
            # liberamos aquí. Si fue timeout, el callback libera después.
            if future.done():
                try:
                    self._slot.release()
                except RuntimeError:
                    pass

    def _run_pipeline(self, frames_rgb: list[np.ndarray]) -> JobResult:
        timings: dict = {}

        # 1) Extracción (incluye decodificación previa hecha en la API).
        t0 = time.perf_counter()
        feat = self._extractor.extract_sequence(frames_rgb)
        timings["extraction"] = (time.perf_counter() - t0) * 1000.0

        # 2) Comprobar calidad: si no pasa, NO invocar el clasificador.
        if not feat.quality_pass:
            return JobResult(
                quality_pass=False,
                accepted=False,
                label_id=None,
                confidence=None,
                reason_codes=list(feat.reason_codes),
                feature_spec_hash=feat.feature_spec_hash,
                model_version=self.model_version,
                timings_ms=timings,
            )

        # Advertencias no bloqueantes se conservan pero no provocan rechazo.
        warnings = [c for c in feat.reason_codes if c in NON_BLOCKING_WARNINGS]

        # 3) Formar lote (1, 6, 147) y predecir.
        x = np.asarray(feat.x, dtype=np.float32)
        if x.shape != SEQUENCE_SHAPE:
            # Fallo técnico de contrato del extractor.
            raise ValueError(f"x del extractor con forma {x.shape} != {SEQUENCE_SHAPE}.")
        batch = x[None, ...]

        t1 = time.perf_counter()
        probs = self._predictor.predict(batch)
        timings["inference"] = (time.perf_counter() - t1) * 1000.0

        # 4) Política de aceptación (Acosta).
        t2 = time.perf_counter()
        decision = self._decision.decide(probs)
        timings["decision"] = (time.perf_counter() - t2) * 1000.0

        reason_codes = list(warnings) + list(decision.reason_codes)
        return JobResult(
            quality_pass=True,
            accepted=decision.accepted,
            label_id=decision.label_id,
            confidence=decision.confidence,
            reason_codes=reason_codes,
            feature_spec_hash=feat.feature_spec_hash,
            model_version=self.model_version,
            timings_ms=timings,
        )

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)
