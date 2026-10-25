# Plan de implementación — Módulo de datos y características (Ortiz, lote 1)

Fuente de verdad: `.agents/tasks/design.md` (APROBADO, `design-review.json` =
APPROVED). Este plan NO rehace diseño; ordena su implementación por DO con
commits atómicos. Reglas de propiedad: `AGENTS.md`. Ortiz SOLO crea/edita
`contracts/`, `src/lsc_data/`, `src/lsc_features/`, `configs/data/`,
`tests/data/`, `tests/features/`, `tests/contracts/`, `reports/data/`,
`docs/data/`, `docs/article/data.md`, `deployment/requirements-data.in`.
PROHIBIDO tocar `app/`, `api/` (solo lectura), resto de `deployment/`,
`tests/app`, `tests/e2e`, `docs/app`, `pyproject.app.toml`, `pytest.ini`, ni
crear `pyproject.toml` raíz.

Comando canónico de pruebas de Ortiz (anula `testpaths` de `pytest.ini` de Juan
pasando paths explícitos, §22/§23):

```
python -m pytest tests/data tests/features tests/contracts
```

## Hallazgos de entorno (verificados en el repo)

- Rama activa y limpia: `ortiz/do01-nucleo-datos-features` (confirmado;
  `git status` solo muestra `.agents/` sin seguimiento).
- Instalados: numpy 2.2.3, pydantic 2.12.5, jsonschema 4.23.0, pytest 9.1.1.
- FALTAN: `PyYAML` (yaml), `opencv-python-headless` (cv2), `mediapipe`.
- Python real = 3.13.2 (el diseño ancla 3.11). Impacto: la suite verde NO debe
  importar cv2/mediapipe en recolección (§8.2/§16: el extractor no carga nada en
  import). PyYAML SÍ se necesita para leer configs en CLI y para las pruebas que
  lean YAML. `test_requirements_version.py` (§20/§22) debe comprobar que el
  intérprete es 3.11 **o** registrar la discrepancia como advertencia/incidente
  sin romper la suite verde (decisión: la prueba verifica pydantic v2 y la
  recolectabilidad de los 3 dirs; la verificación estricta de 3.11 se marca
  `skip`/warning cuando el entorno es 3.13, con mensaje accionable, para no
  fabricar un verde falso ni un rojo ajeno al código de Ortiz).

## Paso 0 — Confirmar rama (primer paso de implementación)

```
git -C c:\Users\danie\Documents2\LSC checkout ortiz/do01-nucleo-datos-features
git -C c:\Users\danie\Documents2\LSC status
```

Instalar dependencias del lote antes de ejecutar pruebas que toquen YAML:

```
python -m pip install -r c:\Users\danie\Documents2\LSC\deployment\requirements-data.in
```

(cv2/mediapipe quedan instalados pero NO se ejercitan en la suite verde.)

---

## DO01 — Contratos (`contracts/`) — commit atómico 1

- [ ] 1. Serialización canónica y hashing (fuente única, §6).
      Crea `c:\Users\danie\Documents2\LSC\contracts\__init__.py` y
      `c:\Users\danie\Documents2\LSC\contracts\canonical.py` con
      `canonical_bytes(obj)` (`json.dumps(ensure_ascii=False, sort_keys=True,
      separators=(",",":"), allow_nan=False)`), `sha256_hex(obj)`, y helpers
      `feature_spec_hash` (§6.1, excluye `build_date`/`*_hash`), `sample_id`
      (§6.2), `sequence_sha256` (§6.3), `manifest_sha256` (§6.4).
      Files: `contracts/__init__.py`, `contracts/canonical.py`
      Verify: `python -m pytest tests/contracts/test_feature_spec_hash.py` (tras DO01-tests) pasa; de inmediato
      `python -c "from contracts.canonical import sha256_hex; print(sha256_hex({'b':1,'a':2}))"` imprime un hex estable.

- [ ] 2. Modelos pydantic v2 del contrato (§4).
      Crea `c:\Users\danie\Documents2\LSC\contracts\models.py` con
      `LabelEntry`, `LabelMap` (10 en orden canónico fijo, validador 0..9
      consecutivos), `FeatureSpec` (§4.2, `features_per_frame=147`,
      `temporal_length=6`, `dtype="float32"`), `ManifestRecord`/`NegativeRecord`/
      `ExclusionRecord` (§4.3, listas de 6, rutas seguras, `class_id` 0..9 o
      null), `SplitAssignment`/`SplitDoc` (§4.4), `ManifestReport`/`SplitReport`/
      `FeatureBuildReport` (§4.5). `ConfigDict(extra="forbid", frozen=True)`.
      Files: `contracts/models.py`
      Verify: `python -c "from contracts.models import FeatureSpec, LabelMap, ManifestRecord; print('ok')"`.

- [ ] 3. Artefactos de contrato serializados (§4.1/§4.2, orden canónico §labels).
      Crea `c:\Users\danie\Documents2\LSC\contracts\labels.json` (10 clases,
      orden `0 Hola..9 Licor` de AGENTS.md; `code` sin tildes, `display_text`
      con tildes, `raw_label`) y `c:\Users\danie\Documents2\LSC\contracts\feature_spec.json`
      (49 nombres `point_order` = pose_0,11,12,13,14,15,16 + lh_0..20 + rh_0..20;
      `feature_spec_hash` calculado con §6.1).
      Files: `contracts/labels.json`, `contracts/feature_spec.json`
      Verify: `python -c "import json; d=json.load(open('contracts/labels.json',encoding='utf-8')); assert len(d['labels'])==10; print('ok')"`.

- [ ] 4. JSON Schemas Draft 2020-12 (§5).
      Crea `contracts/schemas/manifest.schema.json`, `split.schema.json`
      (`seed` const 20261022), `report.schema.json` (`oneOf` por `report_type`),
      `feature_spec.schema.json` (`features_per_frame` const 147,
      `temporal_length` const 6, `dtype` const float32),
      `labels.schema.json` (10 entries). Todos `additionalProperties:false`.
      Files: `contracts/schemas/*.json` (5 archivos, rutas absolutas bajo
      `c:\Users\danie\Documents2\LSC\contracts\schemas\`)
      Verify: `python -c "import json,glob; [json.load(open(p,encoding='utf-8')) for p in glob.glob('contracts/schemas/*.json')]; print('ok')"`.

- [ ] 5. Fixtures sintéticos rotulados (§3, §25).
      Crea `contracts/fixtures/README.md` (declara TODO sintético),
      `SYNTHETIC_valid_sample.jsonl`, `SYNTHETIC_invalid_samples.jsonl`
      (rotulados por `reason_code`), `SYNTHETIC_splits.json`, y
      `SYNTHETIC_feature_result_6x147.npz` (generado con `np.savez` sin pickle,
      `x` float32 (6,147)). Incluye un pequeño script repetible o genera el .npz
      con un snippet documentado en el README.
      Files: `contracts/fixtures/*` (rutas absolutas bajo
      `c:\Users\danie\Documents2\LSC\contracts\fixtures\`)
      Verify: `python -c "import numpy as np; d=np.load('contracts/fixtures/SYNTHETIC_feature_result_6x147.npz'); assert d['x'].shape==(6,147) and d['x'].dtype==np.float32; print('ok')"`.

Commit 1: `feat(contracts): tipos pydantic, hashing canónico, schemas y fixtures sintéticos (DO01)`.

---

## DO01b — `src/lsc_data/` — commit atómico 2

- [ ] 6. Errores tipados y re-export de hashing (§18, §3).
      Crea `src/lsc_data/__init__.py` (exporta API pública: `build_manifest`,
      `make_splits`, etc.), `src/lsc_data/errors.py` (excepciones internas),
      `src/lsc_data/hashing.py` (re-export de `contracts.canonical`).
      Files: `src/lsc_data/__init__.py`, `errors.py`, `hashing.py`
      Verify: `python -c "import src.lsc_data; print('ok')"`.

- [ ] 7. Adaptador de gramática + reconstrucción de secuencias (§9).
      Crea `src/lsc_data/adapter_lsc70w.py` (regex nombrada de `lsc70w.yaml`,
      grupos `signer_id`/`class_label`/`repetition_id`/`frame_index`; falla con
      mensaje accionable si no casa) y `src/lsc_data/sequence.py` (agrupar por
      `(source,signer_id,class_id,repetition_id)`, ordenar por `frame_index`, 6
      índices únicos, excluir faltantes/duplicados/mezclas; nunca rellenar).
      Files: `src/lsc_data/adapter_lsc70w.py`, `sequence.py`
      Verify: cubierto por D01..D05 (`python -m pytest tests/data/test_manifest_D.py`).

- [ ] 8. Manifiesto, inventario, splits (§4.3, §10).
      Crea `src/lsc_data/manifest.py` (`build_manifest -> ManifestReport`,
      `exclusions.csv`), `src/lsc_data/inventory.py` (-> `reports/data/inventory.csv`),
      `src/lsc_data/split.py` (`make_splits(seed=20261022)`,
      `rng=numpy.random.default_rng(20261022)`, `n_test=ceil(0.20N)`,
      `n_val=max(1,floor(0.10N+0.5))` half-up, controles de fuga por identidad y
      por `sequence_sha256`, cobertura de 10 clases -> incidente).
      Files: `src/lsc_data/manifest.py`, `inventory.py`, `split.py`
      Verify: `python -m pytest tests/data/test_manifest_D.py tests/data/test_split_leakage.py`.

- [ ] 9. Adquisición segura sin red (§7) + empaquetado (§17).
      Crea `src/lsc_data/acquisition.py` (`acquire`; sin `--archive` no descarga,
      exit 0 si solo verifica, exit 2 config inválido; con `--archive` valida
      procedencia y extrae), `src/lsc_data/security.py` (temp dir; rechaza
      `..`/absolutas/symlinks que escapan/ejecutables -> exit 3), y
      `src/lsc_data/packaging.py` (`features.npz` tipado `X,y,sample_id,signer_id,
      split,quality_pass`, sin pickle; `feature_spec.json`; `exclusions.csv`).
      Files: `src/lsc_data/acquisition.py`, `security.py`, `packaging.py`
      Verify: `python -m pytest tests/data/test_acquisition_security.py`.

- [ ] 10. Validación y CLI (§18, §19).
      Crea `src/lsc_data/validate.py` (`validate --bundle`; schema/formas/
      integridad/cobertura/fuga; códigos 0/2/3), `src/lsc_data/cli.py`
      (subcomandos `acquire/inventory/manifest/split/features/validate`, mapeo
      0/2/3, `--resume` compara `feature_spec_hash` + `manifest_sha256`, aborta
      exit 3 si difieren) y `src/lsc_data/__main__.py`.
      Files: `src/lsc_data/validate.py`, `cli.py`, `__main__.py`
      Verify: `python -m pytest tests/data/test_cli_exit_codes.py` y
      `python -m lsc_data --help` (lista subcomandos).

Commit 2: `feat(lsc_data): adquisición segura, manifiesto, splits, empaquetado y CLI (DO01)`.

---

## DO01c — `src/lsc_features/` — commit atómico 3

- [ ] 11. Tipos, detector inyectable y códigos de razón (§8.1, §8.2, §13/§18).
      Crea `src/lsc_features/__init__.py` (exporta `FeatureExtractor`,
      `FeatureResult`, `build_features`, `augment_train` — SIN importar cv2/
      mediapipe en el import), `types.py` (`FeatureResult` dataclass frozen
      idéntico en forma a `api/adapters/types.py`, `valid_shoulders:int`),
      `detector.py` (`DetectorProtocol` runtime_checkable, `FakeDetector`,
      `RealDetector` documentado NO ejercitado, tipos `PoseLandmark/PoseResult/
      HandLandmark/HandsResult`), `reason_codes.py` (espejo del frozenset de
      `api/constants.QUALITY_REASON_CODES`, orden de emisión de §13).
      Files: `src/lsc_features/__init__.py`, `types.py`, `detector.py`, `reason_codes.py`
      Verify: `python -c "import src.lsc_features; print('ok')"` (no importa cv2/mediapipe).

- [ ] 12. Matemática pura: geometría, asociación, calidad, temporal (§11–§15).
      Crea `geometry.py` (px, mediana centro/escala comunes a 6 cuadros,
      normalización, `GEOMETRY_OUTLIER` si `|v|>10`, canal `m` máscara binaria),
      `association.py` (asociación mano-pose por costo normalizado por hombros
      del cuadro; precondición de hombros válidos por cuadro §11; ramas 0/1/2
      manos con umbral `<=0.6` y gap `<0.05`), `quality.py` (reglas y orden
      determinista de `reason_codes`), `temporal.py` (`sample_indices` half-up
      con `math.floor(k*(n-1)/5+0.5)`).
      Files: `src/lsc_features/geometry.py`, `association.py`, `quality.py`, `temporal.py`
      Verify: `python -m pytest tests/features/test_features_F.py -k "F03 or F05 or F08 or F09 or F11"`.

- [ ] 13. Extractor, augmentation y build (§8.3, §14, §16).
      Crea `extractor.py` (`FeatureExtractor(config: Mapping, model_paths:
      Mapping[str,str|Path], detector=None)`; acepta dict de feature_spec;
      `model_paths` claves EXACTAS `{"pose","hands"}` -> `CONFIG_MISMATCH` en
      `init()` si falta/sobra; `extract_sequence` garantiza `x (6,147) float32`
      aun con `quality_pass=False`), `augment.py` (`augment_train(x,seed,policy)`
      train_v1: rotación ±5° prob 0.5 + ruido std 0.005 prob 0.5; solo donde
      `m=1`), `build.py` (`build_features(manifest,splits,extractor,out_dir) ->
      FeatureBuildReport`, procesa por `sample_id`, un fallo no detiene).
      Files: `src/lsc_features/extractor.py`, `augment.py`, `build.py`
      Verify: `python -m pytest tests/features/test_features_F.py` (F01,F02,F04,F06,F07,F10,F12 incluidos).

Commit 3: `feat(lsc_features): extractor inyectable, geometría, asociación, calidad, temporal y augment (DO01)`.

---

## DO01d — Pruebas (`tests/`) — commit atómico 4

- [ ] 14. Pruebas de datos D01..D12 (§20).
      Crea `tests/data/__init__.py`, `conftest.py`,
      `test_manifest_D.py` (D01..D11 según §20; D06/D09 son fallo deliberado:
      afirman detección de copia cruzada y fuga de identidad),
      `test_split_leakage.py` (refuerzos D08/D09/D10),
      `test_acquisition_security.py` (D12: exit 3 rutas peligrosas, exit 2 config
      inválido, sin estado parcial), `test_cli_exit_codes.py` (códigos 0/2/3).
      Files: `tests/data/*`
      Verify: `python -m pytest tests/data`.

- [ ] 15. Pruebas de características F01..F12 (§20).
      Crea `tests/features/__init__.py`, `conftest.py` (fixtures de
      `FakeDetector` + geometría sintética), `test_features_F.py` con F01..F12
      incluyendo F08a/b/c/d y F09a/b/c. Fallos deliberados (F08d,F10,F11,F12)
      expresados como expectativa de rechazo/incidente/diferencia.
      Files: `tests/features/*`
      Verify: `python -m pytest tests/features`.

- [ ] 16. Pruebas de contrato (§20).
      Crea `tests/contracts/__init__.py`, `test_feature_spec_hash.py`
      (estable ante rename/fecha, cambia ante `point_order`),
      `test_schema_pydantic_parity.py`, `test_labels_canonical_order.py`,
      `test_incompatible_config.py` (labels/hash divergentes -> fallo al inicio;
      `.task` faltante -> mensaje accionable sin descarga),
      `test_model_paths_contract.py` (claves `{"pose","hands"}`, dict config,
      `CONFIG_MISMATCH` sin tocar disco con FakeDetector),
      `test_requirements_version.py` (existe/parseable/ancla versiones; pydantic
      v2; los 3 dirs recolectables por paths explícitos; Python 3.11 se verifica
      con skip/warning si el entorno difiere, sin romper el verde).
      Files: `tests/contracts/*`
      Verify: `python -m pytest tests/contracts`.

Commit 4: `test(data,features,contracts): suite D01..D12, F01..F12 y contratos (DO01)`.

---

## DO01e — Config, deployment, docs — commit atómico 5

- [ ] 17. Configuración versionada (§7, §9, §11).
      Crea `configs/data/source.yaml` (URL/DOI `10.17632/9ssyn8tff5.2`/versión/
      hashes esperados, extensiones permitidas), `configs/data/lsc70w.yaml`
      (regex nombrada + `label_map`, `repetition_id=0` por defecto),
      `configs/data/features_v1.yaml` (`quality_rules.use_presence=false`,
      umbrales 0.5, política augment train_v1, `detector_versions`/
      `model_asset_hashes` vacíos pendientes DO05).
      Files: `configs/data/*.yaml`
      Verify: `python -c "import yaml,glob; [yaml.safe_load(open(p,encoding='utf-8')) for p in glob.glob('configs/data/*.yaml')]; print('ok')"`.

- [ ] 18. Dependencias y reports (§17, §22, §3).
      Crea `deployment/requirements-data.in` (numpy, opencv-python-headless,
      mediapipe, pydantic>=2, jsonschema, PyYAML, pytest; versiones ancladas),
      `reports/data/.gitkeep`, `reports/data/README.md`.
      Files: `deployment/requirements-data.in`, `reports/data/.gitkeep`,
      `reports/data/README.md`
      Verify: `python -m pytest tests/contracts/test_requirements_version.py`.

- [ ] 19. Documentación (§24, §25) y verificación de `.gitignore` (§21).
      Crea `docs/data/README.md` (comando canónico), `docs/data/progreso-DO.md`
      (bitácora de cierre por DO), `docs/data/corpus-grammar-assumption.md`,
      `docs/data/blocks.md` (DO02/DO05/DO07/DO08/DO09/DO10/DO11 con desbloqueo
      exacto), `docs/article/data.md` (estructura sin métricas inventadas).
      NO añadir patrones a `.gitignore` (a lo sumo un comentario); verificar que
      `contracts/`, fixtures sintéticos y código NO caen bajo exclusión.
      Files: `docs/data/*`, `docs/article/data.md`
      Verify: `git -C c:\Users\danie\Documents2\LSC check-ignore contracts contracts/fixtures/SYNTHETIC_valid_sample.jsonl src/lsc_data` (no imprime nada => no ignorados) y
      `python -m pytest tests/data tests/features tests/contracts` (suite completa verde).

Commit 5: `chore(configs,docs,deployment): config versionada, requirements-data.in y documentación (DO01)`.

---

## Verificación final del lote

```
python -m pytest tests/data tests/features tests/contracts
```

Debe quedar completamente verde con FakeDetector y fixtures sintéticos, SIN red,
SIN corpus real y SIN `.task` reales.

## BLOQUES ABIERTOS (dependen de corpus o pesos reales; NO se implementan)

- **DO02** Descarga del corpus real (Mendeley DOI `10.17632/9ssyn8tff5.2`).
- **DO05-real** Ejecución con `.task` reales de MediaPipe (`RealDetector`).
- **DO07** Paquete de características sobre datos reales.
- **DO08** Paridad con la API de Juan (`valid_shoulders` int↔bool; `labels.json`
  de Acosta; conjunto de `reason_codes`).
- **DO09** Captura externa y consentimientos.
- **DO10** Artículo con métricas reales de Acosta.
- **DO11** Fusión de `testpaths` a la CI raíz (solicitud a Acosta).

Cada bloque queda documentado en `docs/data/blocks.md` con instrucciones exactas
de desbloqueo (§24). Las aceptaciones que dependen de estos bloques se marcan
como ABIERTAS y NO forman parte del verde del lote 1.
