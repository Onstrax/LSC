# Imagen de la aplicación completa (API + interfaz + extractor MediaPipe + modelo Keras).
# Integrada por Acosta sobre la base de Juan (SDD §14).
#
# - Usuario NO root, sistema de archivos de solo lectura (ver compose.yaml), un solo trabajador.
# - Uvicorn escucha 0.0.0.0:8000 DENTRO del contenedor; compose lo publica solo en 127.0.0.1.
# - La cámara permanece en el navegador del anfitrión: NO se monta /dev/video.
# - Los artefactos (model.keras, .task, manifest...) se MONTAN en /artifacts en solo lectura;
#   no se hornean en la imagen. Se generan con `python -m src.lsc_models export-artifacts`.
# - Python 3.11: el conjunto de dependencias se verificó con `pip install --dry-run` para 3.11.
#
# Construir (desde la raíz del repo):  docker compose -f deployment/compose.yaml build

FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app \
    LSC_MODE=real \
    LSC_ARTIFACTS_DIR=/artifacts \
    LSC_HOST=0.0.0.0 \
    LSC_PORT=8000 \
    TF_CPP_MIN_LOG_LEVEL=3 \
    HOME=/tmp \
    KERAS_HOME=/tmp/.keras

# curl: healthcheck. libgl1/libglib2.0-0: las ruedas de OpenCV (dependencia de MediaPipe) los cargan.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 10001 appuser
WORKDIR /app

# Dependencias primero (capa cacheable). pyproject declara solo dependencias (packages = []).
COPY pyproject.toml /app/pyproject.toml
RUN pip install --upgrade pip \
    && pip install ".[serve]"

# Código: API y UI (Juan), extractor (Ortiz), Predictor/DecisionPolicy y capas Keras (Acosta), contratos.
COPY api/ /app/api/
COPY app/ /app/app/
COPY src/ /app/src/
COPY contracts/ /app/contracts/
COPY deployment/ /app/deployment/

RUN mkdir -p /artifacts && chown -R appuser:appuser /app /artifacts
USER appuser

EXPOSE 8000

# TensorFlow tarda en cargar: periodo de arranque holgado.
HEALTHCHECK --interval=15s --timeout=5s --start-period=90s --retries=5 \
    CMD curl -fsS http://127.0.0.1:8000/api/v1/health || exit 1

CMD ["python", "-m", "uvicorn", "api.main:app", \
     "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--no-use-colors"]
