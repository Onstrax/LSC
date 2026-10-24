# Sección del artículo · Arquitectura, despliegue y resultados de uso

> **Estado:** plantilla con la arquitectura y el método ya implementados. Las
> cifras de latencia, usabilidad y las validaciones humanas se completan con
> **ejecuciones reales** (ensayo y piloto). Cada cifra debe apuntar a un archivo
> en `reports/app/`. No se escriben números que no provengan de una medición.
>
> Entrega prevista: 24 de noviembre de 2026. Esta sección corresponde a J5.

## 1. Arquitectura de la aplicación

La aplicación es local y se ejecuta en un contenedor Linux. Sirve, desde el
**mismo origen**, una interfaz web sin framework y una API FastAPI:

```
Navegador (cámara, captura, muestreo)
   │  multipart/form-data: 6 JPEG + metadata (lsc.http.v1)
   ▼
FastAPI  ──►  validación y decodificación JPEG→RGB
   │          trabajador único (una plaza)
   ├────────► Extractor (Ortiz)   FeatureExtractor.extract_sequence → x (6,147), quality_pass
   ├────────► Predictor (Acosta)  Predictor.predict → probabilidades (1,10)
   └────────► Decisión (Acosta)   DecisionAdapter.decide → aceptación, clase, confianza
   ▼
sobre JSON uniforme (accepted / retry / error) + timings_ms
```

El flujo de servidor respeta un orden fijo: validar petición, decodificar,
extraer, comprobar calidad y hash, formar el lote `(1,6,147)`, predecir, aplicar
la política de aceptación y serializar. Si `quality_pass=false`, no se invoca el
clasificador. La aplicación **no** reinterpreta la normalización, los pesos, las
clases ni los umbrales de los módulos vecinos.

## 2. Captura reproducible

El navegador solicita solo video (`audio=false`), resolución ideal 640 × 480 y
objetivo 30 FPS, y usa `requestVideoFrameCallback` para asociar cada muestra a
un cuadro presentado. Con `N` cuadros efectivos (`N ≥ 6`) se seleccionan seis
índices mediante `floor(k·(N−1)/5 + 0,5)` para `k = 0…5`. La vista previa se
espeja solo por CSS; los bytes enviados **no** van espejados. El protocolo
inicial usa 2 000 ms de preparación y 1 000 ms de captura, registrados en
`capture_protocol_version`.

## 3. Despliegue

- Imagen base Python 3.11 (sujeta a la prueba de compatibilidad del proyecto con
  MediaPipe y TensorFlow/Keras), usuario no root, Uvicorn con un trabajador y sin
  reload.
- Compose publica `127.0.0.1:8000:8000` (solo loopback). Modelos y configuración
  se montan en solo lectura; los temporales son efímeros.
- La inferencia ordinaria funciona sin Internet con los recursos preparados; la
  construcción puede requerir red.
- `release_app.json` registra commit, entorno, versión de imagen y hashes de
  artefactos.

## 4. Medición de latencia (método)

El tiempo integral se mide desde el fin de la ventana de captura (antes de
seleccionar y codificar) hasta que el resultado queda presentado en el
navegador, con `performance.now()` en ambos extremos. El servidor registra
`decode`, `extraction`, `inference`, `decision` y `server_total` con un reloj
monotónico propio. Los relojes de navegador y servidor **no** se restan entre
sí. El percentil se define por rango más próximo: `ceil(p·N)`, posiciones desde
1. Los timeouts se reportan censurados.

### Resultados (pendiente de ejecución)

| Métrica | Valor | Fuente |
| --- | --- | --- |
| N (intentos válidos) | _por medir_ | `reports/app/latency_summary.json` |
| p50 integral (ms) | _por medir_ | `reports/app/latency_summary.json` |
| p95 integral de `accepted` (ms) | _por medir_ | `reports/app/latency_summary.json` |
| Tasa de intentos válidos sin fallo técnico | _por medir_ | `reports/app/latency_summary.json` |
| Arranque en frío hasta readiness (s) | _por medir_ | ensayo |

Metas: `p95 < 2 s` y al menos 95 % de intentos válidos sin fallo técnico. La
exactitud y la cobertura del reconocimiento se calculan aparte con Acosta.

## 5. Evaluación de uso (método)

Cinco participantes externos realizan el flujo principal (activar, capturar,
consultar, confirmar/corregir, repetir, limpiar). Se registra por tarea el éxito
sin ayuda, la ayuda necesaria, el tiempo y las incidencias, con un identificador
opaco. Meta: al menos cuatro de cinco completan el flujo sin asistencia.
Cualquier intervención del facilitador durante la tarea cuenta como ayuda.

### Resultados (pendiente de ejecución)

| Métrica | Valor | Fuente |
| --- | --- | --- |
| Participantes que completan sin ayuda | _por medir_ | `reports/app/usability_tasks.csv` |
| Incidencias observadas | _por medir_ | `reports/app/usability_tasks.csv` |

## 6. Límites y amenazas a la validez

- El piloto es exploratorio y no demuestra accesibilidad para toda la comunidad
  sorda.
- Los metadatos por sí solos no prueban cómo se capturó el video; la prueba
  física de lados y el test de navegador son obligatorios.
- La confirmación del usuario no es una etiqueta validada científicamente ni se
  usa para reentrenar.
- Si las metas de rendimiento no se alcanzan, se reportan las mediciones y el
  análisis; no se sustituyen cifras.

## 7. Versiones y hashes

Ver la tabla de compatibilidad en
[`version-compatibility.md`](../app/version-compatibility.md) y `release_app.json`.
