# Pipeline real de datos y features LSC70W (lote 2, Ortiz)

La rama `ortiz/do01-nucleo-datos-features` activa el camino REAL del módulo de datos sobre el corpus LSC70W (39 firmantes) y los `.task` de MediaPipe: reescribe la gramática al layout real (FEAT-001), implementa `RealDetector.detect_pose/detect_hands` sobre MediaPipe Tasks y registra la procedencia de los `.task` (FEAT-002), cablea la orquestación real de `run_features` + `--raw-dir` para producir el bundle `v1` (FEAT-003), y cierra la paridad de contrato con la API más la documentación/reporte con conteos reales (FEAT-004). El cambio es consistente con el reparto de propiedad: toca solo rutas de Ortiz. Los conteos del reporte se trazan a sus fuentes y verifican de forma independiente (inventario 39/2323/388, hash canónico estable, parity + orquestación en verde).

Watch for: una sola observación menor, no bloqueante — el reporte afirma "88 passed, 1 skipped" para la suite canónica completa, cifra que no re-ejecuté por instrucción (confirmed que la evidencia la registra; possible que el entorno difiera, sin impacto en el veredicto). Todo lo demás verifica limpio.

**Verdict**: APPROVED

## Vista de alto nivel

La gramática pasó de un patrón supuesto (`S\d{3}/[a-z]+/frame_N.jpg`) al layout real `Per##/CLASE/Per##_CLASE_K.jpg` con clases en MAYÚSCULAS y `label_map` crudo→code. La clase canónica es la carpeta de primer nivel; el regex relaja deliberadamente solo el token de clase del *nombre de archivo* (no la carpeta) para absorber irregularidades reales del corpus (carpeta BUENAS con archivos `BUENOS`, doble guion bajo, guion bajo final), sin alterar `class_id` ni el orden canónico 0..9.

Los conteos son reales y trazables. El inventario tiene 39 filas `Per##` que suman 2323 `.jpg` y 388 ejecuciones; la única anomalía (`Per39/NOMBRE` con 1 frame) se EXCLUYE con `INVALID_FRAME_COUNT` y no se rellena, dejando 387 muestras válidas. Las clases TARDES/YO quedan en 38 por ausencia física en Per39 (8 subdirs, no 10), y NOMBRE en 38 por la exclusión. Es una limitación real documentada, no un defecto.

El `feature_spec.json` registra SHA-256 reales de los dos `.task` y recomputa el `feature_spec_hash` canónico; recalculado en vivo coincide con el almacenado, y `features_v1.yaml` refleja los mismos valores para que config y contrato no diverjan.

La paridad de contrato (DO08) fija la compatibilidad estructural de `FeatureResult` entre Ortiz y la API en modo solo-lectura, documenta la única divergencia (`valid_shoulders` int→bool) y la coacción que aplica la API, y comprueba `labels.json` contra `api/constants` y `api/schemas` sin tocar `api/`.

La propiedad se respeta estrictamente: ningún archivo bajo `app/`, `api/`, `pytest.ini`, `pyproject*.toml`, lockfile ni CI raíz cambia; no hay push ni force; no se versionan `.task` ni datos pesados.

<details>
<summary>Issues (1)</summary>

1. **Conteo de suite no re-verificado** — el reporte cita "88 passed, 1 skipped" para la suite canónica completa; por diseño del paso no se re-ejecutó la suite entera (sí el subconjunto parity+orquestación, 11 passed). No bloqueante: la evidencia lo registra y el spot-check en verde es consistente.

</details>

<details>
<summary>Details</summary>

## Gramática real y relajación controlada del regex

El cambio en `configs/data/lsc70w.yaml` reescribe `path_regex` a `^(?P<signer_id>Per\d{2})/(?P<class_label>[A-Z]+)/(?P=signer_id)_[A-Z]+_+(?P<frame_index>\d+)_?\.jpg$`. El punto sutil, y bien razonado, es que la clase canónica se deriva de la **carpeta** (grupo `class_label` → `label_map`), nunca del token de clase del nombre de archivo. Por eso la retro-referencia `(?P=signer_id)` se mantiene estricta sobre el firmante, pero el token de clase del archivo se afloja a `[A-Z]+` para tolerar la carpeta `BUENAS` cuyos archivos se llaman `BUENOS` (38 firmantes), el doble guion bajo `CLASE__<n>` (6 archivos) y el guion bajo final `<n>_` (4 archivos). Como el token de clase del archivo es `[A-Z]+` sin dígitos, `frame_index` (`\d+`) no es ambiguo. El `label_map` queda en MAYÚSCULAS (`HOLA:0 … LICOR:9`) y `contracts/labels.json` no se toca, preservando el orden canónico. El findings de FEAT-001 reporta 0 archivos sin casar sobre los 2323, lo que respalda que la relajación cubre el corpus completo.

## Trazabilidad de conteos a su fuente

Verifiqué el inventario de forma independiente: `reports/data/inventory.csv` tiene 39 filas `Per##`, suma 2323 en `n_files` y 388 en `n_subdirs`, exactamente los valores 39/2323/388 del reporte. La aritmética de exclusión cierra: 38 firmantes × 10 clases + Per39 × 8 clases = 388 ejecuciones; 388 − 1 exclusión (`Per39/NOMBRE`, `INVALID_FRAME_COUNT`, 1 frame) = 387 válidas. La cobertura por clase `{3 TARDES:38, 5 YO:38, 6 NOMBRE:38, resto:39}` es coherente: Per39 tiene 8 subdirectorios (no 10), así que TARDES y YO faltan físicamente y NOMBRE cae por la exclusión. La anomalía del único `.jpg` se excluye explícitamente y no se rellena — tal como exige el criterio. El split de personas 27/4/8 suma 39 con 0 incidentes, y el bundle 267/40/80 suma 387. Son conteos reales, no fabricados.

## Procedencia de los .task y estabilidad del feature_spec_hash

`contracts/feature_spec.json` pasa de `detector_versions`/`model_asset_hashes` en `null` a valores concretos: pose `4eaa5eb7…c4ad`, hands `fbc2a300…7cde`, con versiones `mediapipe-tasks:1.2.2:…`. Recalculé el hash canónico en vivo (`contracts.canonical.feature_spec_hash`) sobre el JSON y coincide con el almacenado `08779533…4c004`, confirmando que el `feature_spec_hash` es estable y consistente con el contenido actual (incluye los hashes de `.task`, como exige el diseño §19/§24). `configs/data/features_v1.yaml` refleja idénticos `detector_versions` y `model_asset_hashes`, evitando divergencia config↔contrato. Los SHA-256 reales de los `.task` no pude recomputarlos (los `.task` no están versionados, por diseño), pero su registro y el encadenamiento al hash canónico sí están verificados.

```
stored     08779533e99cc466db5845498789e13b11bcab02144d41135915a1e24961c004
recomputed 08779533e99cc466db5845498789e13b11bcab02144d41135915a1e24961c004  (match)
```

## RealDetector sobre MediaPipe Tasks, imports perezosos

`src/lsc_features/detector.py` reemplaza los `NotImplementedError` de `detect_pose`/`detect_hands` por envolturas reales de MediaPipe Tasks: `_to_mp_image` construye el `mp.Image(SRGB)` con import perezoso, y los `detect_*` traducen `pose_landmarks`/`hand_landmarks` a los dataclasses de Ortiz, devolviendo resultados vacíos cuando MediaPipe no detecta. El `handedness` se transporta pero no interviene en la asociación (consistente con design §11). Los imports de `mediapipe` permanecen perezosos (dentro de `__init__` y los `detect_*`), de modo que importar el paquete no carga MediaPipe — un requisito para que la suite corra sin la dependencia pesada. No se tocó la matemática del extractor (geometry/association/quality) ni umbrales.

## Orquestación real del bundle e inyección para pruebas

`run_features` deja de abortar con `ConfigError` y ahora: carga el `feature_spec` canónico desde `contracts/feature_spec.json` (fuente del hash que exige `validate`), toma `quality_rules` de `features_v1.yaml`, construye el `RealDetector` sobre `artifacts/models/*.task`, recorre el manifiesto ordenado por `sample_id`, extrae cada secuencia y persiste el bundle con columna `split` mapeada desde `splits.json`, `feature_spec.json` y `exclusions.csv`. Dos decisiones importantes y correctas: si falta un `.task` en el camino real lanza `ConfigError` accionable (exit 2) **sin fallback a mock**, cumpliendo la regla 3 de AGENTS.md; y un fallo por muestra se registra en `failed_sample_ids` y no detiene el proceso, pero si no se extrae ninguna muestra el bundle vacío se bloquea con `IntegrityError` (exit 3). El `--raw-dir` nuevo en la CLI es opcional en la firma pero obligatorio en el camino real (se valida), sin tocar otros subcomandos ni `pytest.ini`.

El `detector` y el `frame_loader` son parámetros opcionales que existen para inyección en pruebas; por defecto se usa el camino real. Esto permite que `tests/data/test_features_build_orchestration.py` ejercite la orquestación completa con un `FakeDetector` y un loader sintético (6 arreglos RGB uint8), verificando que el bundle resultante pasa `validate` (exit 0), tiene columna `split` no vacía coherente con `splits.json` y `X` de forma `(N,6,147)` float32. Ejecuté ese test como spot-check acotado y pasa, confirmando el criterio de aceptación de FEAT-003 sin re-correr el pipeline real.

## Paridad de contrato DO08

`tests/contracts/test_feature_result_parity.py` fija la compatibilidad estructural entre `src.lsc_features.types.FeatureResult` y `api.adapters.types.FeatureResult`: mismos nombres de campo y en el mismo orden (`x, quality_pass, reason_codes, valid_shoulders, frames_with_hand, feature_spec_hash, elapsed_ms`). Documenta la única divergencia — `valid_shoulders` anotado `int` en Ortiz vs `bool` en la API — y reproduce la coacción `bool(getattr(result, "valid_shoulders", True))`, que además verifica leyendo literalmente `api/adapters/real.py` (confirmado: la línea existe en `api/adapters/real.py:80`). La prueba también comprueba igualdad de conjuntos de `reason_codes` (9 códigos) y que `labels.json` (10 entradas, `class_id` 0..9 consecutivos) es coherente con `NUM_CLASSES` y con el rango `label_id` leído de `PredictResponse.model_fields`. Todo el acceso a `api/` y `app/` es solo lectura; no se construyen ni mutan objetos de `api/`.

## Propiedad de rutas y seguridad de git

El conjunto de archivos modificados vs `main` cae enteramente en rutas de Ortiz: `contracts/feature_spec.json`, `configs/data/{features_v1,lsc70w}.yaml`, `src/lsc_data/{cli,features_build}.py`, `src/lsc_features/detector.py`, `tests/contracts/`, `tests/data/`, `reports/data/`, `docs/article/data.md`, `docs/data/progreso-DO.md` y la evidencia bajo `.agents/tasks/`. El filtro explícito por `app/`, `api/`, `pytest.ini`, `pyproject.toml`, `pyproject.app.toml`, locks y `.github/` devuelve vacío: ninguno se tocó. No hay `.task`, `.jpg` ni `.npz` del bundle versionados (`git ls-files` solo lista `artifacts/README.md` y un fixture sintético preexistente en `contracts/fixtures/`). La rama está `ahead 4` de `origin` sin push, y el reflog no muestra force. Los artefactos pesados (`data/interim`, `artifacts/data/v1`) quedan git-ignorados por diseño.

## Limitaciones declaradas, no bloqueantes

El reporte deja explícitos los bloques pendientes que pertenecen a otros responsables o a captura física: DO09 (captura externa y consentimientos), DO11 (fusión de `testpaths` al CI raíz de Acosta — Ortiz correctamente no edita `pytest.ini`), y las métricas del MODELO (Acosta), que `docs/article/data.md` deja como pendiente sin inventar cifras. La cobertura reducida de algunas clases en splits (consecuencia del tamaño real del corpus y la ausencia en Per39) está documentada con conteos y es una limitación legítima, no un motivo de rechazo.

</details>

<details>
<summary>Mapa de archivos</summary>

- `configs/data/lsc70w.yaml` — gramática real `Per##/CLASE/...`, `label_map` MAYÚSCULAS, regex relajado solo en el nombre de archivo.
- `configs/data/features_v1.yaml` — `detector_versions`/`model_asset_hashes` reales (espejo del contrato).
- `contracts/feature_spec.json` — SHA-256 reales de los `.task` y `feature_spec_hash` recomputado.
- `src/lsc_features/detector.py` — `RealDetector.detect_pose/detect_hands` reales con imports perezosos.
- `src/lsc_data/cli.py` — nuevo `--raw-dir` en el subcomando `features`.
- `src/lsc_data/features_build.py` — orquestación real de `run_features` (detector/loader inyectables, sin fallback a mock).
- `tests/contracts/test_feature_result_parity.py` — paridad estructural FeatureResult y labels/reason_codes vs API (solo lectura).
- `tests/data/test_features_build_orchestration.py` — bundle válido con `FakeDetector` + loader sintético.
- `reports/data/inventory.csv`, `reports/data/do-batch2-report.md` — conteos reales con fuente.
- `docs/article/data.md`, `docs/data/progreso-DO.md` — cierre documental con conteos reales, sin métricas de modelo.
- `.agents/tasks/task-do-batch2-ortiz/features/FEAT-00{1..4}.json` — evidencia de verificación por FEAT.

Diff completo: `git diff main` en `ortiz/do01-nucleo-datos-features`.

</details>
