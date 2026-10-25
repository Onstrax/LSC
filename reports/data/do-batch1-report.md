# Reporte de cierre — Lote 1 del módulo de datos (Ortiz)

Resumen del estado del lote 1 (FEAT-001..005) del módulo de datos y extracción de
características. Sin red, sin corpus real, sin `.task` reales: toda la geometría y
el pipeline se validan con detectores falsos inyectables y fixtures sintéticos
(design §23/§25).

## Entregables de FEAT-005

- `configs/data/features_v1.yaml` — reglas de calidad/umbrales (espejo de
  `contracts/feature_spec.json`), política de augment `train_v1`,
  `detector_versions`/`model_asset_hashes` pendientes (bloque DO05).
- `configs/data/source.yaml`, `configs/data/lsc70w.yaml` — procedencia y
  gramática del corpus (ya existentes; sin cambios en este paso).
- `deployment/requirements-data.in` — 7 dependencias ancladas, entregadas a
  Acosta (ya existente; sin cambios).
- `reports/data/.gitkeep`, `reports/data/README.md`, `reports/data/do-batch1-report.md`.
- `docs/data/README.md` (existente), `docs/data/progreso-DO.md`,
  `docs/data/corpus-grammar-assumption.md`, `docs/data/blocks.md`.
- `docs/article/data.md` — estructura de la sección de datos, sin métricas
  inventadas (bloque DO10).

## Verificación (comando canónico de paths explícitos)

```
python -m pytest tests/data tests/features tests/contracts -q
```

Resultado: 77 passed, 1 skipped. El único skip es la comprobación de versión de
Python (entorno 3.13 ≠ 3.11 anclado por el diseño); no rompe el verde y la
congelación a 3.11 corresponde a la fusión al lock raíz de Acosta.

La suite NO ejercita `cv2`/`mediapipe` (el extractor no carga detectores en el
import; detector inyectable + fixtures sintéticos, design §8.2/§16). `mediapipe`
no ofrece el pin 0.10.21 para Python 3.13, por lo que no se instala en este
entorno; el pin se entrega como solicitud a Acosta sin modificarlo.

## Estado de los criterios de aceptación de FEAT-005

- Los 3 YAML de `configs/data` cargan con `yaml.safe_load`;
  `features_v1.yaml.quality_rules.use_presence == False`. ✔
- `requirements-data.in` existe, lista 7 dependencias ancladas, consumido por
  `test_requirements_version.py`. ✔
- `git check-ignore --no-index` no reporta como ignorados `contracts`, los
  fixtures sintéticos, `src/lsc_data`, `configs/data/*`, `reports/data/*`,
  `docs/data/*`, `tests/data/*` ni `deployment/requirements-data.in` (tras el
  anclaje `data/`→`/data/` autorizado por el usuario). ✔
- `docs/data/blocks.md` documenta los 7 bloques (DO02, DO05, DO07, DO08, DO09,
  DO10, DO11) con desbloqueo exacto. ✔
- Suite verde con el comando canónico. ✔

## Bloques ABIERTOS (NO completados; dependen de corpus/.task/artefactos externos)

- **DO02** — Descarga del corpus real (sin red en este lote).
- **DO05** — Ejecución con `.task` real de MediaPipe (camino real no ejercitado).
- **DO07** — Paquete de características sobre datos reales (depende de DO02+DO05).
- **DO08** — Paridad con la API de Juan (`valid_shoulders` int↔bool, `labels.json`
  de Acosta).
- **DO09** — Captura externa y consentimientos.
- **DO10** — Métricas del modelo de Acosta en `docs/article/data.md`.
- **DO11** — Fusión de `testpaths` al `pyproject.toml` raíz (CI de Acosta).

Condiciones de desbloqueo exactas en `docs/data/blocks.md`.
