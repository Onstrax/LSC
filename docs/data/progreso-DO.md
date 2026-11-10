# Bitácora de cierre por DO (módulo de datos, Ortiz)

Registro por tarea DO del lote 1: id, cambio, archivo(s) de salida, comando de
verificación, conteos medidos y riesgo/bloque pendiente (design §25). Los
conteos de pruebas provienen del comando canónico
`python -m pytest tests/data tests/features tests/contracts`.

## DO01 — Configuración versionada, dependencias, reports y docs (FEAT-005)

- **Cambio.** Configuración del extractor v1, solicitud de dependencias a Acosta,
  carpeta de reportes y documentación del módulo; verificación de `.gitignore`.
- **Archivos de salida.** `configs/data/features_v1.yaml` (además de
  `source.yaml` y `lsc70w.yaml` ya existentes), `deployment/requirements-data.in`,
  `reports/data/.gitkeep`, `reports/data/README.md`,
  `reports/data/do-batch1-report.md`, `docs/data/README.md`,
  `docs/data/progreso-DO.md`, `docs/data/corpus-grammar-assumption.md`,
  `docs/data/blocks.md`, `docs/article/data.md`.
- **Comando de verificación.**
  ```
  python -c "import yaml,glob; cfgs={__import__('os').path.basename(p): yaml.safe_load(open(p,encoding='utf-8')) for p in glob.glob('configs/data/*.yaml')}; assert cfgs['features_v1.yaml']['quality_rules']['use_presence'] is False; print('configs ok')"
  git check-ignore contracts contracts/fixtures/SYNTHETIC_valid_sample.jsonl src/lsc_data deployment/requirements-data.in
  python -m pytest tests/data tests/features tests/contracts
  ```
- **Conteos medidos.** 3 YAML en `configs/data/` cargan con `yaml.safe_load`;
  `features_v1.yaml.quality_rules.use_presence == False`. `requirements-data.in`
  lista 7 dependencias ancladas. Suite: 77 passed, 1 skipped (skip por entorno
  Python 3.13 != 3.11 anclado; no rompe el verde).
- **Riesgo/bloque.** CERRADO. El patrón `data/` del `.gitignore` raíz (sin
  anclar) ignoraba CUALQUIER carpeta `data` a cualquier profundidad, incluidas
  `configs/data/`, `reports/data/`, `docs/data/` y `tests/data/`, impidiendo
  versionar los archivos nuevos. El diseño §21 asumía —incorrectamente— que
  `data/` sólo cubría el árbol de datos pesados. **Corrección acordada y
  autorizada por el usuario** (dueño de la decisión de propiedad): el usuario
  ancló el patrón a la raíz `data/`→`/data/` en el `.gitignore` raíz. Verificado
  con `git check-ignore --no-index`: `configs/data/*`, `reports/data/*`,
  `docs/data/*` y `tests/data/*` quedan versionables, mientras `data/raw`,
  `data/interim`, `data/processed` y `artifacts/data` siguen ignorados. Ortiz NO
  reclama propiedad de `.gitignore` (archivo raíz, fuera de sus rutas); el cambio
  lo hizo el usuario, no este módulo.

## DO03/DO04 — Extractor y suite (contexto, lote 1)

Implementados en features previas (código del extractor y suite D01..D12 /
F01..F12). Verificación cubierta por el mismo comando canónico de pruebas. Las
ejecuciones DO03/DO04 sobre el corpus REAL se registran abajo (lote 2).

---

# Lote 2 — Corpus real, `.task` reales y cierre (task-do-batch2-ortiz)

Entradas por DO del lote 2 (FEAT-001/FEAT-002/FEAT-003/FEAT-004). Conteos REALES
medidos sobre el corpus `LSC70W` (39 firmantes). Cada número cita su fuente.

## DO02 — Adquisición/verificación del corpus (FEAT-001)

- **Cambio.** Reescritura de la gramática `configs/data/lsc70w.yaml` al corpus
  real `Per##/CLASE/Per##_CLASE_K.jpg` (clases en MAYÚSCULAS) y verificación de
  procedencia sin red.
- **Archivos de salida.** `configs/data/lsc70w.yaml` (gramática/`label_map`),
  procedencia de `configs/data/source.yaml`.
- **Comando de verificación.**
  ```
  python -m src.lsc_data acquire --config configs/data/source.yaml
  ```
- **Conteos medidos.** `acquire` exit 0, verificación sin red, DOI
  `10.17632/9ssyn8tff5.2`, 0 archivos movidos. El adapter parsea
  `Per01/HOLA/Per01_HOLA_0.jpg` → `Per01 HOLA 0 0`; los 2323 `.jpg` casan la
  gramática (0 sin casar).
- **Riesgo/bloque.** La retro-referencia estricta `(?P=class_label)` no casaba
  por irregularidades SÓLO en el nombre de archivo (carpeta `BUENAS` con
  archivos `Per##_BUENOS_*`; doble guion bajo; guion final). Se relajó SÓLO la
  parte del nombre, manteniendo la carpeta como `class_label` (fuente del
  `class_id`); semántica del manifiesto sin cambios. No queda bloque de DO02.

## DO03 — Inventario y manifiesto sobre el corpus real (FEAT-001)

- **Cambio.** Ejecución de `inventory` y `manifest` sobre `LSC70W`.
- **Archivos de salida.** `reports/data/inventory.csv`,
  `data/interim/manifest.jsonl`, `data/interim/exclusions.csv`
  (git-ignorados salvo documentación).
- **Comando de verificación.**
  ```
  python -m src.lsc_data inventory --raw-dir "Speak in Colombian Sign Language A Dynamic LSC70 Database/LSC70/LSC70/LSC70W" --out reports/data/inventory.csv
  python -m src.lsc_data manifest --config configs/data/lsc70w.yaml --raw-dir "Speak in Colombian Sign Language A Dynamic LSC70 Database/LSC70/LSC70/LSC70W" --out data/interim/manifest.jsonl --exclusions data/interim/exclusions.csv
  ```
- **Conteos medidos.** `inventory.csv`: 39 grupos `Per##` (Per01 `60,10` …
  Per39 `43,8`). `manifest.jsonl`: **387 válidas**; `exclusions.csv`: **1
  excluida** (`lsc70_v2/Per39/6/0`, Per39/NOMBRE, `INVALID_FRAME_COUNT`). Por
  clase (válidas): `{0:39, 1:39, 2:39, 3:38, 4:39, 5:38, 6:38, 7:39, 8:39,
  9:39}`.
- **Riesgo/bloque.** CERRADO. Per39 sin TARDES/YO y NOMBRE con 1 cuadro quedan
  documentados, no rellenados.

## DO04 — Splits por firmante con semilla fija (FEAT-001)

- **Cambio.** `split` por `signer_id` con semilla `20261022`.
- **Archivos de salida.** `data/interim/splits.json` (git-ignorado).
- **Comando de verificación.**
  ```
  python -m src.lsc_data split --manifest data/interim/manifest.jsonl --seed 20261022 --out data/interim/splits.json
  ```
- **Conteos medidos.** `seed 20261022`; **train/val/test = 27/4/8 personas**
  (suma 39); **incidentes = 0**; sin fuga de identidad (exit 0). Fuente:
  `data/interim/splits.json`.
- **Riesgo/bloque.** La semilla NO se recomputa ante cobertura incompleta.

## DO05 — Camino REAL de MediaPipe y procedencia de `.task` (FEAT-002)

- **Cambio.** Implementación de `RealDetector.detect_pose/detect_hands` en
  `src/lsc_features/detector.py` (mapeo MediaPipe Tasks → dataclasses de Ortiz,
  imports perezosos) y registro de procedencia de los `.task` con recomputo del
  `feature_spec_hash` canónico.
- **Archivos de salida.** `src/lsc_features/detector.py`,
  `contracts/feature_spec.json`, `configs/data/features_v1.yaml`.
- **Comando de verificación.**
  ```
  python -c "from src.lsc_features.detector import RealDetector; import cv2; d=RealDetector('artifacts/models/pose_landmarker.task','artifacts/models/hand_landmarker.task'); img=cv2.cvtColor(cv2.imread('Speak in Colombian Sign Language A Dynamic LSC70 Database/LSC70/LSC70/LSC70W/Per01/HOLA/Per01_HOLA_0.jpg'),cv2.COLOR_BGR2RGB); print('poses',len(d.detect_pose(img).pose_landmarks),'hands',len(d.detect_hands(img).hand_landmarks)); d.close()"
  python -c "import json; from contracts.canonical import feature_spec_hash; s=json.load(open('contracts/feature_spec.json',encoding='utf-8')); print('hash_ok', feature_spec_hash(s)==s['feature_spec_hash'])"
  ```
- **Conteos medidos.** Sobre `Per01/HOLA/Per01_HOLA_0.jpg`: poses 1, hands 1,
  sin `NotImplementedError`. Procedencia (mediapipe 1.2.2):
  `detector_versions.pose='mediapipe-tasks:1.2.2:pose_landmarker.task'` (y hands
  análogo); `model_asset_hashes.pose='4eaa5eb7…c4ad'`,
  `.hands='fbc2a300…607cde'` (SHA-256 reales confirmados con `Get-FileHash`).
  `feature_spec_hash` recomputado →
  `08779533e99cc466db5845498789e13b11bcab02144d41135915a1e24961c004`
  (`feature_spec_hash(spec)==stored` True). Import del paquete NO carga
  mediapipe (`mediapipe_loaded` False).
- **Riesgo/bloque.** Modo real por defecto; si falta un `.task`, el camino real
  no disponible (sin fallback a mock, AGENTS.md regla 3).

## DO07 — Paquete de características REAL (FEAT-003)

- **Cambio.** `src/lsc_data/features_build.py::run_features` reescrito de un
  `ConfigError` incondicional a la orquestación real (RealDetector sobre los
  `.task`, `FeatureExtractor`, `frame_loader` contra `--raw-dir`,
  `packaging.write_bundle` con columna split y `feature_spec.json`). CLI
  `features` acepta `--raw-dir`.
- **Archivos de salida.** `src/lsc_data/features_build.py`,
  `src/lsc_data/cli.py`, `tests/data/test_features_build_orchestration.py`,
  bundle `artifacts/data/v1/{features.npz,feature_spec.json,exclusions.csv}`
  (git-ignorado).
- **Comando de verificación.**
  ```
  python -m src.lsc_data features --config configs/data/features_v1.yaml --manifest data/interim/manifest.jsonl --splits data/interim/splits.json --raw-dir "Speak in Colombian Sign Language A Dynamic LSC70 Database/LSC70/LSC70/LSC70W" --out artifacts/data/v1
  python -m src.lsc_data validate --bundle artifacts/data/v1
  ```
- **Conteos medidos.** `features` exit 0 (~112 s); `validate` exit 0. Fuente
  `artifacts/data/v1/features.npz`: **N=387**, forma `(6,147)` `float32`;
  **splits train 267 / val 40 / test 80** (muestras); **`quality_pass` True 371
  / False 16**; rechazos por clase `{0:2, 1:1, 2:1, 3:5, 4:4, 5:1, 9:2}`. N ==
  válidas del manifiesto.
- **Riesgo/bloque.** Un fallo por muestra se registra sin detener el lote; sin
  fallback a mock.

## DO08 — Paridad de contrato con la API de Juan (FEAT-004)

- **Cambio.** Prueba de contrato que fija la compatibilidad estructural del
  `FeatureResult` (nombres de campo, divergencia documentada
  `valid_shoulders` int↔bool con la coacción `bool(...)` que aplica
  `api/adapters/real.py`), la igualdad del conjunto `reason_codes` y la
  coherencia de `contracts/labels.json` con `api/constants` (`NUM_CLASSES`) y el
  esquema HTTP (`api/schemas/models.PredictResponse.label_id`). SÓLO LECTURA de
  `app/` y `api/`.
- **Archivos de salida.** `tests/contracts/test_feature_result_parity.py`.
- **Comando de verificación.**
  ```
  python -m pytest tests/contracts/test_feature_result_parity.py -q
  ```
- **Conteos medidos.** 10 passed. No importa ni modifica `api/` salvo
  `api.adapters.types`, `api.constants` y `api.schemas.models` (lectura).
- **Riesgo/bloque.** Ninguno; la unificación vive del lado de Ortiz sin tocar
  `api/`.

## DO10 — Cierre documental con conteos reales (FEAT-004)

- **Cambio.** `docs/article/data.md` con los conteos REALES (39 personas, 388
  ejecuciones, 2323 imágenes, 387 válidas / 1 excluida, splits 27/4/8, bundle
  N=387 con train/val/test 267/40/80 y quality_pass 371/16); esta bitácora; y
  `reports/data/do-batch2-report.md`. Sin métricas del modelo de Acosta.
- **Archivos de salida.** `docs/article/data.md`, `docs/data/progreso-DO.md`,
  `reports/data/do-batch2-report.md`.
- **Comando de verificación.**
  ```
  python -m pytest tests/data tests/features tests/contracts -q
  python -c "import json,collections; recs=[json.loads(l) for l in open('data/interim/manifest.jsonl',encoding='utf-8') if l.strip()]; d=json.load(open('data/interim/splits.json',encoding='utf-8')); print('validas',len(recs),'tvt',len(d['signer_ids']['train']),len(d['signer_ids']['val']),len(d['signer_ids']['test']))"
  ```
- **Conteos medidos.** Suite canónica: 88 passed, 1 skipped. Chequeo cruzado:
  `validas 387 tvt 27 4 8`. Los conteos del artículo/reporte coinciden con
  `reports/data/inventory.csv`, `data/interim/manifest.jsonl`,
  `data/interim/splits.json` y `artifacts/data/v1`.
- **Riesgo/bloque.** Pendientes DO09 (captura externa), DO11 (fusión de
  `testpaths` al CI raíz de Acosta) y métricas del modelo de Acosta (DO10 del
  artículo de resultados).

## Cierre del lote 1 DO01 (task-do-batch1-ortiz)

- **Id.** `task-do-batch1-ortiz` — estado `completed`. Revisión
  `.agents/tasks/do-batch1-review.json` con verdict **APPROVED** (2 NIT sin
  acción requerida: ancla `.gitignore` como decisión del dueño y skip previsto
  por entorno Python 3.13 != 3.11).
- **Cambio.** Cierre del lote 1 del módulo de datos y extracción de Ortiz
  (FEAT-001..FEAT-005): contratos, `src/lsc_data`, `src/lsc_features`,
  `configs/data`, suite D01..D12 / F01..F12 + contratos, `requirements-data.in`
  y documentación. Sin red, sin corpus real, sin `.task` reales (detector
  inyectable y fixtures sintéticos).
- **Archivos del cierre.** `.agents/tasks/task-do-batch1-ortiz/task.json`
  (status→`completed`), `docs/data/progreso-DO.md` (esta bitácora),
  `docs/data/blocks.md` (nota de cierre del `.gitignore`),
  `reports/data/do-batch1-report.md`, junto con los entregables ya versionados
  en `configs/data/`, `reports/data/`, `docs/data/` y `docs/article/data.md`.
- **Comando de verificación.**
  ```
  python -m pytest tests/data tests/features tests/contracts
  ```
- **Conteos de pruebas.** Suite canónica **VERDE**: 78 ítems recolectados,
  **77 passed, 1 skipped** (el skip es por diseño: entorno Python 3.13 != 3.11
  anclado; se activará al fusionar el lock raíz de Acosta, bloque DO11). Sin
  fallos.
- **Bloques abiertos (fuera del lote 1, documentados en `docs/data/blocks.md`).**
  - **DO02** — Descarga del corpus real. Desbloquea al colocar el archivo local
    y correr `acquire` (verifica DOI/versión/tamaño/SHA-256); nunca descarga por
    red.
  - **DO05** — Ejecución con `.task` reales de MediaPipe. Desbloquea al obtener
    los `.task` con procedencia verificable y registrar versiones/hashes en
    `features_v1.yaml`.
  - **DO07** — Paquete de características sobre datos reales. Depende de DO02 y
    DO05.
  - **DO08** — Paridad con la API de Juan (unificar `FeatureResult`,
    `reason_codes` y `labels.json`). Los cambios en `api/` los hace Juan.
  - **DO09** — Captura externa y consentimientos. Requiere protocolo de captura;
    nunca se versionan copias de consentimiento.
  - **DO10** — Artículo final con métricas de Acosta. Las métricas reales entran
    cuando Acosta las entregue; hoy sin métricas inventadas.
  - **DO11** — Fusión de `testpaths` en la config de pruebas raíz. Se solicita a
    Acosta (dueño de la CI raíz); Ortiz no modifica `pytest.ini`.
