# Contrato HTTP · `lsc.http.v1`

Documento legible del contrato que propone Juan desde `api/schemas/`. La versión
acordada se congela en [`../../api/schemas/openapi.json`](../../api/schemas/openapi.json)
y Ortiz la incorpora en `contracts/`, con una prueba que detecte divergencias.
**No se mantienen dos contratos activos distintos.**

Base: mismo origen que la interfaz. Versión fija `schema_version = lsc.http.v1`.

## Endpoints

| Método | Ruta | Propósito |
| --- | --- | --- |
| POST | `/api/v1/predict` | Enviar un intento (6 JPEG + metadata) y recibir sugerencia/estado |
| GET | `/api/v1/health` | Readiness y versiones |
| GET | `/api/v1/config` | Etiquetas, protocolo de captura, límites y capacidades para la UI |
| GET | `/api/v1/live` | Liveness del proceso |

## POST /api/v1/predict

`multipart/form-data` con:

- **Seis** campos repetidos `frames`, en orden temporal, nombrados
  `frame_0.jpg` … `frame_5.jpg`.
- Un campo de texto `metadata` con JSON validado (no una segunda petición).

### Campos de `metadata`

| Campo | Tipo y regla |
| --- | --- |
| `schema_version` | Cadena fija `lsc.http.v1` |
| `client_request_id` | UUID nuevo por intento; solo correlación, sin identidad personal |
| `capture_protocol_version` | Versión exacta anunciada por `/config` |
| `source` | `webcam` o `authorized_fixture` (el segundo queda marcado en la evidencia) |
| `width`, `height` | Enteros positivos iguales a las dimensiones JPEG decodificadas |
| `frame_count` | Entero `6` |
| `observed_frame_count` | Entero `N ≥ 6` de cuadros muestreados antes de la selección |
| `selected_indices` | Seis enteros que cumplen `floor(k·(N−1)/5 + 0,5)`, rango `0…N−1` |
| `timestamps_ms` | Seis números finitos estrictamente crecientes, relativos al inicio |
| `capture_duration_ms` | Duración real de la ventana; incluye los timestamps seleccionados |
| `mirrored` | Booleano `false` |

### Límites

| Límite | Valor |
| --- | --- |
| Cuerpo multipart completo | 6 MiB |
| Cada JPEG | 512 KiB |
| `metadata` | 16 KiB |
| Dimensión máxima por imagen | 640 × 480 (307 200 px) |

La API verifica que cada imagen sea un JPEG decodificable (el nombre y el
`Content-Type` no bastan) y que las seis compartan dimensiones.

## Respuesta: sobre JSON uniforme

Toda respuesta generada por la aplicación usa el mismo sobre, **incluidos los
errores de validación**. Los campos anulables permanecen presentes con `null`.

| Campo | Regla |
| --- | --- |
| `schema_version`, `request_id` | `lsc.http.v1` y UUID del servidor |
| `client_request_id` | UUID de la solicitud o `null` si no se pudo validar |
| `status` | `accepted`, `retry` o `error` |
| `label_id`, `display_text` | Clase `0…9` y texto de `labels.json` solo en `accepted`; `null` en otros |
| `confidence` | Probabilidad de la clase elegida en `accepted`; `null` si no hay sugerencia |
| `reason_codes` | Lista ordenada; puede contener advertencias no bloqueantes |
| `model_version`, `feature_spec_hash` | Versiones verificadas; `null` si no se cargaron |
| `capture_protocol_version` | Versión efectiva o `null` si la solicitud es inválida |
| `mode` | `real` o `mock` (visible en desarrollo y exportaciones) |
| `timings_ms` | `decode`, `extraction`, `inference`, `decision`, `server_total`; `null` por etapa no ejecutada |

### Significado de los estados

- **accepted**: el sistema ofrece una sugerencia según su política. **No**
  significa que una persona haya confirmado su corrección.
- **retry**: captura procesada que requiere repetición por calidad o confianza.
- **error**: falla técnica o solicitud inválida.

## Códigos HTTP

| HTTP | Situación | `status` |
| --- | --- | --- |
| 200 | `accepted` o `retry` (control de calidad o umbral) | accepted / retry |
| 400 | `metadata` JSON ilegible o multipart mal formado | error |
| 413 | Cuerpo o archivo demasiado grande | error |
| 415 | Formato distinto de JPEG | error |
| 422 | Contrato, cantidad, dimensiones o valores inválidos | error |
| 429 | Hay una inferencia activa (`SERVICE_BUSY`, con `Retry-After`) | error |
| 503 | Modelo no disponible o configuración incompatible | error |
| 504 | Se agotó el plazo de procesamiento (`INFERENCE_TIMEOUT`) | error |
| 500 | Excepción inesperada o salida inválida del modelo (`MODEL_OUTPUT_INVALID`) | error |

No se devuelven trazas, rutas privadas ni detalles de configuración al usuario
final. El cliente no reenvía automáticamente capturas fallidas: cada repetición
requiere una acción visible del usuario.

## Códigos de razón (`reason_codes`)

De **calidad** (propiedad de Ortiz): `INVALID_FRAME_COUNT`, `INVALID_IMAGE`,
`SIZE_MISMATCH`, `MULTIPLE_PEOPLE`, `INSUFFICIENT_POSE`, `INSUFFICIENT_HANDS`,
`HAND_ASSOCIATION_AMBIGUOUS` (puede ser advertencia si `quality_pass=true`),
`GEOMETRY_OUTLIER`, `CONFIG_MISMATCH`.

De **la API / política**: `INVALID_METADATA`, `MALFORMED_MULTIPART`,
`BODY_TOO_LARGE`, `FILE_TOO_LARGE`, `UNSUPPORTED_MEDIA_TYPE`,
`CONTRACT_VIOLATION`, `DIMENSION_MISMATCH`, `MIRROR_DECLARED`, `SERVICE_BUSY`,
`MODEL_UNAVAILABLE`, `CONFIG_INCOMPATIBLE`, `INFERENCE_TIMEOUT`,
`MODEL_OUTPUT_INVALID`, `INTERNAL_ERROR`, `LOW_CONFIDENCE`.

## Ejemplos sintéticos

Ver [`../../api/fixtures/`](../../api/fixtures/):
`response_accepted.json`, `response_retry.json`, `response_error.json`.

## Regenerar el OpenAPI

```bash
python deployment/export_openapi.py
```
