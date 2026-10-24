# Tabla de compatibilidad de versiones y hashes

Este documento consolida las versiones del módulo de Juan y los artefactos de
los componentes vecinos. Las celdas marcadas _por acordar_ se completan al
integrar con Ortiz y Acosta y al generar `release_app.json`.

## Contrato y protocolo (propiedad de Juan)

| Elemento | Valor |
| --- | --- |
| `schema_version` (HTTP) | `lsc.http.v1` |
| `capture_protocol_version` | `lsc.capture.v1` |
| `api_version` | `0.1.0` |
| Preparación / ventana de captura | 2 000 ms / 1 000 ms |
| Forma de secuencia | `(6, 147)` — 49 puntos × (x, y, presencia) |

## Entorno de ejecución

| Elemento | Valor objetivo | Verificado localmente |
| --- | --- | --- |
| Python (contenedor) | 3.11 (acordado; sujeto a prueba de compatibilidad) | 3.14 en el equipo de desarrollo |
| FastAPI | ver `requirements-app.txt` | 0.162.0 |
| Pydantic | ver `requirements-app.txt` | 2.14.1 |
| Pillow | ver `requirements-app.txt` | 13.0.0 |
| NumPy | ver `requirements-app.txt` | 2.5.4 |
| Uvicorn | ver `requirements-app.txt` | instalado |

> Las versiones de MediaPipe y TensorFlow/Keras las fija Acosta en el lock
> compartido tras la prueba de compatibilidad. No se introducen aquí para evitar
> un segundo entorno incompatible.

## Artefactos del modelo (propiedad de Acosta)

| Artefacto | Versión / hash | Estado |
| --- | --- | --- |
| `model.keras` (`model_version`) | _por acordar_ | pendiente de entrega |
| `labels.json` (orden canónico) | 10 clases | contrato fijado |
| `feature_spec.json` (`feature_spec_hash`) | _por acordar_ | debe coincidir con el del extractor |
| `thresholds.json` (`policy_version`) | _por acordar_ | requerido para habilitar predicción real |
| `manifest.json` | _por acordar_ | hashes verificables |

## Artefactos del extractor (propiedad de Ortiz)

| Artefacto | Versión / hash | Estado |
| --- | --- | --- |
| `pose_landmarker.task` | _por acordar_ | pendiente de entrega |
| `hand_landmarker.task` | _por acordar_ | pendiente de entrega |
| `feature_spec_hash` del extractor | _por acordar_ | debe coincidir con el del modelo |

## Orden canónico de etiquetas (fuente única: `labels.json`)

| id | etiqueta |
| --- | --- |
| 0 | Hola |
| 1 | Buenos |
| 2 | Días |
| 3 | Tardes |
| 4 | Noches |
| 5 | Yo |
| 6 | Nombre |
| 7 | Años |
| 8 | Gustar |
| 9 | Licor |

## Generación automática

`release_app.json` consolida commit, entorno y hashes reales:

```bash
python deployment/make_release.py --image lsc-app:local --artifacts ./artifacts
```
