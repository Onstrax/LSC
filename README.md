# LSC · Reconocimiento de señas aisladas — Módulo de Juan Esteban

Aplicación web local que captura una **seña aislada** de la Lengua de Señas
Colombiana (LSC), la envía a un servicio FastAPI, coordina el **extractor de
características** (Ortiz) y el **predictor** (Acosta), y presenta una sugerencia
de etiqueta comprensible al usuario.

Este repositorio contiene el **30 % del proyecto** que corresponde a Juan
Esteban Álvarez García: interfaz, API, despliegue, pruebas de uso, medición de
latencia y evidencia para el artículo y la exposición. El reconocimiento en sí
(extractor y modelo) lo aportan los componentes vecinos a través de contratos
versionados; este módulo **no** modifica su normalización, pesos, clases ni
umbrales.

> Vocabulario admitido (10 clases de LSC70W, orden canónico):
> `0 Hola · 1 Buenos · 2 Días · 3 Tardes · 4 Noches · 5 Yo · 6 Nombre ·
> 7 Años · 8 Gustar · 9 Licor`

## Estructura del repositorio

```
app/                 Interfaz web (HTML/CSS/JS sin framework)
  index.html         Pantalla principal accesible
  styles.css         Diseño accesible
  js/                Estado, cámara, muestreo, cliente HTTP, resultados, métricas
api/                 Aplicación FastAPI
  schemas/           Propuesta del contrato HTTP (OpenAPI congelado)
  fixtures/          Respuestas sintéticas rotuladas (accepted/retry/error)
  adapters/          Extractor, predictor y decisión (reales + dobles de prueba)
deployment/          Dockerfile, compose.yaml, scripts de arranque y empaquetado
tests/
  app/               Pruebas de API con dobles deterministas (A01–A14)
  e2e/               Pruebas de navegador con Playwright (U01–U15)
docs/
  app/               Guía de usuario, guía del operador, contrato HTTP, fallos
  article/           Sección de despliegue del artículo
  presentation/      Guion y material de demostración
reports/app/         Evidencias de latencia y usabilidad (sin imágenes personales)
```

> **Propiedad de archivos.** Juan es propietario de `app/`, `api/`,
> `deployment/`, `tests/app/`, `tests/e2e/`, `docs/app/`,
> `docs/article/deployment.md`, `reports/app/` y `docs/presentation/`.
> `contracts/`, `tests/contracts/` y `src/lsc_features/` pertenecen a **Ortiz**;
> `src/lsc_inference/` y los artefactos del modelo pertenecen a **Acosta**.
> Ver `AGENTS.md`.

## Inicio rápido (desarrollo, modo simulado)

```bash
# 1. Crear y activar entorno virtual (Python 3.11 objetivo)
python -m venv .venv
source .venv/bin/activate      # fish: source .venv/bin/activate.fish

# 2. Instalar dependencias de la aplicación (sin modelo real)
pip install -r deployment/requirements-app.txt

# 3. Arrancar en modo de prueba (respuestas simuladas)
LSC_MODE=mock python -m uvicorn api.main:app --host 127.0.0.1 --port 8000

# 4. Abrir la interfaz
#    http://localhost:8000
```

> En modo `mock` la interfaz muestra el rótulo persistente
> **"Modo de prueba · respuestas simuladas"**. El modo por defecto de la
> entrega es `real` y requiere los artefactos de Ortiz y Acosta.

## Despliegue real (contenedor)

```bash
docker compose -f deployment/compose.yaml up --build -d
docker compose -f deployment/compose.yaml ps
python deployment/check_ready.py
```

Consulta `docs/app/operator-guide.md` para la preparación de artefactos y la
resolución de fallos, y `docs/app/user-guide.md` para el uso de la aplicación.

## Pruebas

```bash
python -m pytest tests/app -q        # pruebas rápidas con dobles deterministas
python -m pytest tests/e2e -q        # pruebas de navegador (requiere Playwright)
```

## Estado del contrato HTTP

- Versión del contrato: `lsc.http.v1`
- Endpoint de predicción: `POST /api/v1/predict` (`multipart/form-data`)
- Salud: `GET /api/v1/health` · Configuración de UI: `GET /api/v1/config`
- Especificación congelada: [`api/schemas/openapi.json`](api/schemas/openapi.json)
- Documento legible: [`docs/app/http-contract.md`](docs/app/http-contract.md)

## Licencia y datos

El corpus LSC70W procede de *Speak in Colombian Sign Language* (Mendeley Data,
CC BY 4.0). Este repositorio **no** contiene imágenes personales,
consentimientos ni equivalencias de identidad. Las evidencias de uso se
conservan sin datos personales.
