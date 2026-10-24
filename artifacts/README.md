# Artefactos del modelo y del extractor

Este directorio se monta en el contenedor en **solo lectura** (`/artifacts`).
No forma parte de la imagen y **no se versiona** en Git (ver `.gitignore`).

El operador coloca aquí los artefactos entregados por los compañeros antes de
arrancar el servicio en **modo real**.

## Archivos esperados

De **Acosta** (modelo y política):

| Archivo | Contenido |
| --- | --- |
| `model.keras` | Modelo entrenado |
| `labels.json` | Etiquetas en orden canónico (10 clases) |
| `feature_spec.json` | Especificación de características (incluye `feature_spec_hash`) |
| `preprocessing.json` | Parámetros de preprocesamiento |
| `thresholds.json` | Política de aceptación versionada (`policy_version`, `threshold`, `mode`) |
| `model_card.md` | Tarjeta del modelo |
| `manifest.json` | `model_version`, `feature_spec_hash`, hashes |

De **Ortiz** (extractor):

| Archivo | Contenido |
| --- | --- |
| `pose_landmarker.task` | Modelo de pose de MediaPipe |
| `hand_landmarker.task` | Modelo de manos de MediaPipe |

## Verificación

Tras montar los artefactos y arrancar:

```bash
python deployment/check_ready.py --url http://127.0.0.1:8000
```

Si falta cualquier archivo o hay incompatibilidad de hashes, el servicio
permanece **no disponible** (health 503). Nunca cambia automáticamente a modo
simulado.
