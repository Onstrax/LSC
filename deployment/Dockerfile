# Imagen de la aplicación de Juan (SDD §14).
#
# - Usuario NO root.
# - Uvicorn escucha 0.0.0.0:8000 DENTRO del contenedor, un trabajador, sin reload.
# - Los recursos visuales se distribuyen localmente (sin CDN).
# - La cámara permanece en el navegador del anfitrión: NO se monta /dev/video,
#   no se usa modo privilegiado ni la red del host.
#
# NOTA DE COMPATIBILIDAD: Python 3.11 es el punto de partida acordado, sujeto a
# la prueba de compatibilidad del proyecto con MediaPipe y TensorFlow/Keras.
# La imagen base y las dependencias pesadas (modelo/extractor) se fijan en el
# lock compartido de Acosta tras verificar compatibilidad. Esta imagen instala
# las dependencias de la APLICACIÓN; el lock raíz añade las de inferencia.

FROM python:3.11-slim AS base

# Evitar escritura de .pyc y forzar salida sin buffer para logs legibles.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    LSC_MODE=real \
    LSC_ARTIFACTS_DIR=/artifacts \
    LSC_HOST=0.0.0.0 \
    LSC_PORT=8000

# curl para el healthcheck (sin dependencias de red en ejecución ordinaria).
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Usuario no root.
RUN useradd --create-home --uid 10001 appuser
WORKDIR /app

# Instalar dependencias de la aplicación primero (capa cacheable).
# En el repositorio final, Acosta integra el lock raíz; aquí usamos el
# requirements de la aplicación para una imagen autosuficiente de la UI/API.
COPY deployment/requirements-app.txt /app/deployment/requirements-app.txt
RUN pip install --upgrade pip \
    && pip install -r /app/deployment/requirements-app.txt

# Copiar el código del módulo de Juan.
COPY api/ /app/api/
COPY app/ /app/app/
COPY deployment/ /app/deployment/

# Directorio de artefactos montado en solo lectura por compose.
RUN mkdir -p /artifacts && chown -R appuser:appuser /app /artifacts

USER appuser

EXPOSE 8000

# Healthcheck contra /api/v1/health con periodo de arranque holgado.
HEALTHCHECK --interval=15s --timeout=5s --start-period=40s --retries=5 \
    CMD curl -fsS http://127.0.0.1:8000/api/v1/health || exit 1

# Un solo trabajador, sin reload.
CMD ["python", "-m", "uvicorn", "api.main:app", \
     "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--no-use-colors"]
