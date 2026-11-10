# Reporte de cierre — Lote 2 del módulo de datos (Ortiz)

Resumen del lote 2 (`task-do-batch2-ortiz`, FEAT-001/FEAT-002/FEAT-003/FEAT-004)
del módulo de datos y extracción de características. A diferencia del lote 1
(sintético, sin corpus ni `.task`), este lote ejecuta el camino REAL sobre el
corpus `LSC70W` (39 firmantes) y los `.task` de MediaPipe con procedencia
verificada. Alcance: **clasificación de 10 señas aisladas**, no traducción. Las
métricas del MODELO son de Acosta y quedan pendientes (no se inventan).

## Qué se activó

- **FEAT-001 (DO02/DO03/DO04).** Gramática del corpus real, `acquire` sin red,
  `inventory`, `manifest` y `split` con semilla `20261022`.
- **FEAT-002 (DO05).** `RealDetector.detect_pose/detect_hands` sobre MediaPipe
  Tasks; procedencia de los `.task` (SHA-256) y recomputo del
  `feature_spec_hash`.
- **FEAT-003 (DO07).** Orquestación real de `run_features` + `--raw-dir` en la
  CLI; bundle `artifacts/data/v1` con columna split y `feature_spec.json`.
- **FEAT-004 (DO08/DO10).** Prueba de paridad del contrato con la API y cierre
  documental con conteos reales (este reporte, el artículo y la bitácora).

## Comandos canónicos

```
# Suite (paths explícitos; sobrescribe el testpaths de pytest.ini de Juan sin editarlo)
python -m pytest tests/data tests/features tests/contracts -q

# CLI del módulo de datos (NUNCA 'python -m lsc_data')
python -m src.lsc_data acquire   --config configs/data/source.yaml
python -m src.lsc_data inventory --raw-dir "Speak in Colombian Sign Language A Dynamic LSC70 Database/LSC70/LSC70/LSC70W" --out reports/data/inventory.csv
python -m src.lsc_data manifest  --config configs/data/lsc70w.yaml --raw-dir "Speak in Colombian Sign Language A Dynamic LSC70 Database/LSC70/LSC70/LSC70W" --out data/interim/manifest.jsonl --exclusions data/interim/exclusions.csv
python -m src.lsc_data split     --manifest data/interim/manifest.jsonl --seed 20261022 --out data/interim/splits.json
python -m src.lsc_data features  --config configs/data/features_v1.yaml --manifest data/interim/manifest.jsonl --splits data/interim/splits.json --raw-dir "Speak in Colombian Sign Language A Dynamic LSC70 Database/LSC70/LSC70/LSC70W" --out artifacts/data/v1
python -m src.lsc_data validate  --bundle artifacts/data/v1
```

Códigos de salida de la CLI: 0 éxito/conforme, 2 config inválida, 3 fallo de
integridad.

## Conteos reales (cada número con su fuente)

| Métrica | Valor | Fuente |
| --- | --- | --- |
| Personas (firmantes `Per##`) | 39 | `reports/data/inventory.csv` |
| Ejecuciones (clases × persona) | 388 | `reports/data/inventory.csv` (`n_subdirs`) |
| Imágenes `.jpg` | 2323 | `reports/data/inventory.csv` (`n_files`) |
| Muestras válidas (manifiesto) | 387 | `data/interim/manifest.jsonl` |
| Excluidas | 1 (`Per39/NOMBRE`, `INVALID_FRAME_COUNT`) | `data/interim/exclusions.csv` |
| Split (personas) train/val/test | 27 / 4 / 8 | `data/interim/splits.json` |
| Incidentes de split | 0 | `data/interim/splits.json` |
| Bundle N (muestras) | 387 | `artifacts/data/v1/features.npz` |
| Bundle split (muestras) train/val/test | 267 / 40 / 80 | `artifacts/data/v1/features.npz` |
| `quality_pass` True / False | 371 / 16 | `artifacts/data/v1/features.npz` |
| Forma / dtype | `(6,147)` / `float32` | `artifacts/data/v1/features.npz` |

Cobertura por clase en el manifiesto (válidas), fuente
`data/interim/manifest.jsonl`:
`{0 Hola:39, 1 Buenos:39, 2 Días:39, 3 Tardes:38, 4 Noches:39, 5 Yo:38,
6 Nombre:38, 7 Años:39, 8 Gustar:39, 9 Licor:39}` (Tardes/Yo en 38 por ausencia
en Per39; Nombre en 38 por la exclusión de `Per39/NOMBRE`).

Rechazos de calidad por clase en el bundle (`quality_pass=False`), fuente
`artifacts/data/v1/features.npz`:
`{0 Hola:2, 1 Buenos:1, 2 Días:1, 3 Tardes:5, 4 Noches:4, 5 Yo:1, 9 Licor:2}`.

Procedencia de los `.task` (fuente: `contracts/feature_spec.json`): pose
`4eaa5eb7…c4ad`, hands `fbc2a300…607cde`; `feature_spec_hash`
`08779533e99cc466db5845498789e13b11bcab02144d41135915a1e24961c004`.

## Artefactos generados (git-ignorados por `.gitignore` raíz)

- `reports/data/inventory.csv`
- `data/interim/manifest.jsonl`, `data/interim/exclusions.csv`,
  `data/interim/splits.json`
- `artifacts/data/v1/{features.npz, feature_spec.json, exclusions.csv}`

Versionados en este lote: `tests/contracts/test_feature_result_parity.py`,
`docs/article/data.md`, `docs/data/progreso-DO.md` y este reporte (además del
código de `src/lsc_data`, `src/lsc_features`, `configs/data` tocado en
FEAT-001..003).

## Resultado de la suite

```
python -m pytest tests/data tests/features tests/contracts -q
```

Resultado: **88 passed, 1 skipped**. El único skip es la comprobación de versión
de Python (entorno 3.13 ≠ 3.11 anclado por el diseño; se activará al fusionar el
lock raíz de Acosta, bloque DO11). Sin fallos.

## Bloques restantes

- **DO09 — Captura externa y consentimientos.** Requiere el protocolo de captura
  física; nunca se versionan copias de consentimiento.
- **DO11 — Fusión de `testpaths` al CI raíz de Acosta.** Ortiz no edita
  `pytest.ini` ni el `pyproject.toml` raíz; se solicita a Acosta que incluya
  `tests/data`, `tests/features` y `tests/contracts` en la recolección de CI.
- **Métricas del modelo (Acosta).** Exactitud, matriz de confusión y
  calibración para la sección de resultados del artículo. Hoy `docs/article/
  data.md` no contiene métricas de modelo; se incorporan cuando Acosta las
  entregue.
