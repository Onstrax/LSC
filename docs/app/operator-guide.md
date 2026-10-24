# Guía del operador

El operador instala y arranca el servicio. El usuario final solo interactúa con
el navegador. Esta guía cubre instalación, preparación de artefactos, readiness,
fallos de cámara y recuperación.

## Requisitos

- Docker y Docker Compose en el equipo de demostración.
- Los artefactos del modelo (Acosta) y del extractor (Ortiz) disponibles para
  montarlos en solo lectura.
- Una cámara conectada al equipo que ejecuta el **navegador** (no al contenedor).
- Navegador certificado moderno con `requestVideoFrameCallback` (Chromium
  reciente). `getUserMedia` requiere un contexto seguro; `localhost` lo permite.

## Preparación de artefactos (modo real)

Coloque en `./artifacts` del repositorio los archivos entregados por el equipo:

De **Acosta**: `model.keras`, `labels.json`, `feature_spec.json`,
`preprocessing.json`, `thresholds.json`, `model_card.md`, `manifest.json`.

De **Ortiz**: `pose_landmarker.task`, `hand_landmarker.task` (y la configuración
del extractor).

Detalles de cada archivo en [`../../artifacts/README.md`](../../artifacts/README.md).

> El directorio `artifacts/` se monta en `/artifacts` en **solo lectura** y no
> se versiona en Git.

## Arranque (contenedor, modo real)

Desde la raíz del repositorio:

```bash
# Construye y arranca el servicio real; falla de forma comprensible si faltan
# artefactos (el servicio queda no disponible, no cambia a simulado).
docker compose -f deployment/compose.yaml up --build -d

# Estado y salud
docker compose -f deployment/compose.yaml ps

# Verificación de readiness, modo real, versión y hashes
python deployment/check_ready.py --url http://127.0.0.1:8000
```

Abra el navegador en `http://localhost:8000`.

Para detener:

```bash
docker compose -f deployment/compose.yaml down
```

## Arranque en desarrollo (modo simulado)

Para probar la interfaz sin modelo real:

```bash
pip install -r deployment/requirements-app.txt
LSC_MODE=mock python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

En modo `mock` la interfaz muestra el rótulo persistente **“Modo de prueba ·
respuestas simuladas”**. La demostración final y los reportes científicos
**excluyen** este modo.

## Readiness y salud

`GET /api/v1/health` devuelve `ready`, `busy`, `mode`, `api_version`,
`model_version`, `feature_spec_hash` y `capture_protocol_version`.

- **200** si la configuración está preparada (incluso con `busy=true`).
- **503** si el servicio está cargando, degradado o incompatible.

Si falta un artefacto o hay incompatibilidad de hashes, el servicio **permanece
no disponible**; nunca cambia automáticamente a modo simulado.

El arranque en frío se mide por separado, desde iniciar el servicio hasta la
readiness real.

## Recuperación de fallos

| Síntoma | Causa probable | Acción |
| --- | --- | --- |
| `check_ready.py` falla con 503 | Faltan artefactos o hashes incompatibles | Revise `./artifacts`; compare versiones con Acosta/Ortiz. |
| Respuestas 429 | Hay una inferencia activa | Es normal bajo concurrencia; espere. Si persiste, reinicie. |
| Respuestas 504 | El trabajo superó el plazo | El servicio queda degradado; reinicie el contenedor. |
| La cámara no aparece | Permiso denegado o cámara en uso | Revise permisos del navegador; cierre otras apps de cámara. |
| “Navegador no compatible” | Falta `requestVideoFrameCallback` | Use el navegador certificado. |

Reinicio del contenedor:

```bash
docker compose -f deployment/compose.yaml restart
```

## Notas de seguridad del despliegue

- El servicio se publica **solo en loopback** (`127.0.0.1:8000`). No lo exponga
  en una IP de red como sustituto de `localhost`: el acceso remoto exige HTTPS y
  un diseño adicional fuera del núcleo.
- La cámara permanece en el navegador del anfitrión. No se monta `/dev/video`,
  no se usa modo privilegiado ni la red del host.
- La construcción de la imagen puede requerir red; la **inferencia ordinaria**
  funciona sin Internet con los recursos preparados.
